from __future__ import annotations

import io
import importlib.util
import uuid
import unittest
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from ..app import app
from ..config import DEMO_DATA_DIR, WEBSITE_DIR


class StudioApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        email = f"test-{uuid.uuid4().hex[:10]}@example.org"
        response = self.client.post(
            "/api/auth/login",
            json={"email": email, "access_code": "topas-preview"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["token"]
        self.headers = {"Authorization": f"Bearer {token}"}

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_demo_uses_real_outputs_and_omits_rules(self) -> None:
        response = self.client.get("/api/projects", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        demo = next(project for project in response.json()["projects"] if project["is_demo"])
        detail = self.client.get(f"/api/projects/{demo['id']}", headers=self.headers)
        self.assertEqual(detail.status_code, 200)
        manifest = detail.json()["manifest"]["roles"]
        self.assertIn("macro_actions", manifest["system"]["merged_components"])
        self.assertIn("user_profile", manifest["user"]["merged_components"])
        self.assertNotIn("rules", manifest["system"]["merged_components"])

        component = self.client.get(
            f"/api/projects/{demo['id']}/components/system/macro_actions",
            headers=self.headers,
        )
        self.assertEqual(component.status_code, 200)
        self.assertGreater(len(component.json()["data"]["macro_actions"]), 0)

        blocked_rules = self.client.get(
            f"/api/projects/{demo['id']}/components/system/rules",
            headers=self.headers,
        )
        self.assertEqual(blocked_rules.status_code, 404)

    def test_runtime_assets_are_bundled_with_the_website(self) -> None:
        spec = importlib.util.find_spec("topa")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.origin)
        package_file = Path(str(spec.origin)).resolve()
        self.assertTrue(package_file.is_relative_to(WEBSITE_DIR.resolve()))
        self.assertTrue((DEMO_DATA_DIR / "cbt_showcase" / "extracted_components").is_dir())
        self.assertTrue((DEMO_DATA_DIR / "cbt_showcase" / "refined_components").is_dir())

    def test_components_are_editable_and_refinement_is_available(self) -> None:
        projects = self.client.get("/api/projects", headers=self.headers).json()["projects"]
        demo = next(project for project in projects if project["is_demo"])
        project_id = demo["id"]

        component_url = f"/api/projects/{project_id}/components/system/macro_actions"
        original = self.client.get(component_url, headers=self.headers)
        self.assertEqual(original.status_code, 200, original.text)
        data = original.json()["data"]
        data["macro_actions"][0]["description"] = "Profile-specific edited description"

        saved = self.client.put(
            component_url,
            headers=self.headers,
            json={"phase": "extraction", "data": data},
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(
            saved.json()["data"]["macro_actions"][0]["description"],
            "Profile-specific edited description",
        )
        self.assertEqual(saved.json()["project"]["refinement_status"], "stale")

        refined = self.client.get(
            f"{component_url}?phase=refinement",
            headers=self.headers,
        )
        self.assertEqual(refined.status_code, 200, refined.text)
        report = self.client.get(
            f"/api/projects/{project_id}/refinement/summary",
            headers=self.headers,
        )
        self.assertEqual(report.status_code, 200, report.text)

        invalid_graph = self.client.put(
            f"/api/projects/{project_id}/components/system/knowledge_graph",
            headers=self.headers,
            json={"phase": "extraction", "data": {"nodes": []}},
        )
        self.assertEqual(invalid_graph.status_code, 422)

    def test_create_project_and_upload_one_role(self) -> None:
        response = self.client.post(
            "/api/projects",
            headers=self.headers,
            json={
                "name": "Nutrition coaching blueprint",
                "domain": "Nutrition coaching",
                "system_name": "Coach",
                "user_name": "Client",
                "interaction_unit": "conversation",
                "model": "gpt-5.1",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        project_id = response.json()["project"]["id"]

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Chapter 1 - Foundations")
        content = document.tobytes()
        document.close()
        upload = self.client.post(
            f"/api/projects/{project_id}/books",
            headers=self.headers,
            data={"role": "system"},
            files=[("files", ("coaching-guide.pdf", io.BytesIO(content), "application/pdf"))],
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        project = upload.json()["project"]
        self.assertEqual(len(project["books"]), 1)
        self.assertNotIn("stored_path", project["books"][0])
        self.assertEqual(project["books"][0]["role"], "system")

    def test_access_and_project_validation(self) -> None:
        denied = self.client.post(
            "/api/auth/login",
            json={"email": "person@example.org", "access_code": "wrong"},
        )
        self.assertEqual(denied.status_code, 401)
        unsafe = self.client.post(
            "/api/projects",
            headers=self.headers,
            json={
                "name": "Unsafe",
                "domain": "../outside",
                "system_name": "System",
                "user_name": "User",
                "interaction_unit": "session",
                "model": "gpt-4.1",
            },
        )
        self.assertEqual(unsafe.status_code, 422)


if __name__ == "__main__":
    unittest.main()
