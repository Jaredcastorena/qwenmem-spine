"""Release scans must inspect the intended source copy and fail closed."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "source"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "vendor/candle-core").mkdir(parents=True)
        for name in ("check-release-hygiene.sh", "check-vendored-candle.py"):
            shutil.copyfile(SCRIPTS / name, self.root / "scripts" / name)
        (self.root / "vendor/candle-core-source.json").write_text(
            json.dumps({"files": {}}))
        (self.root / "README.md").write_text("An unpacked release fixture.\n")

    def scan(self, *args):
        return subprocess.run(
            ["bash", str(self.root / "scripts/check-release-hygiene.sh"), *args],
            cwd=self.parent, capture_output=True, text=True, check=False)

    def test_clean_source_copy_from_unrelated_working_directory(self):
        result = self.scan("--source-tree")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_parent_git_repository_is_rejected(self):
        subprocess.run(["git", "init", "--quiet", str(self.parent)], check=True)
        result = self.scan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not its own Git root", result.stderr)

    def test_private_path_reports_filename_without_disclosing_value(self):
        value = str(Path("/").joinpath("home", "example", "private"))
        (self.root / "leak.txt").write_text(value)
        result = self.scan("--source-tree")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("leak.txt", result.stdout)
        self.assertNotIn(value, result.stdout + result.stderr)

    def test_source_link_is_rejected(self):
        (self.root / "external").symlink_to(self.parent)
        result = self.scan("--source-tree")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symbolic link", result.stderr)

    def test_vendor_change_is_rejected(self):
        (self.root / "vendor/candle-core/extra.rs").write_text("// changed\n")
        result = self.scan("--source-tree")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from reviewed source", result.stderr)


if __name__ == "__main__":
    unittest.main()
