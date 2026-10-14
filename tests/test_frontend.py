from pathlib import Path
import unittest

from fastapi.responses import FileResponse, RedirectResponse

from api.main import FRONTEND_DIR, app, frontend, root


class FrontendTests(unittest.TestCase):
    def test_dashboard_and_static_routes_are_registered(self):
        route_paths = {route.path for route in app.routes}

        self.assertIn("/app", route_paths)
        self.assertIn("/static", route_paths)

    def test_root_redirects_to_dashboard(self):
        response = root()

        self.assertIsInstance(response, RedirectResponse)
        self.assertEqual(response.headers["location"], "/app")

    def test_dashboard_assets_exist_and_use_local_api(self):
        response = frontend()
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        javascript = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIsInstance(response, FileResponse)
        self.assertTrue(Path(response.path).is_file())
        self.assertIn('/static/styles.css', html)
        self.assertIn('/static/app.js', html)
        self.assertIn('fetch("/prototype/query"', javascript)


if __name__ == "__main__":
    unittest.main()
