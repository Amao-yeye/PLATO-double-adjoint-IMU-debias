import hashlib
import json
import tarfile

from build_release import build, PREFIX


def test_public_archive_excludes_research_data_and_has_matching_hashes(tmp_path):
    output = tmp_path / "source.tar.gz"
    build(output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert not any("/data/" in name or "/Results/" in name or "/papers/" in name or "/.git/" in name for name in names)
        assert not any(name.endswith((".pt", ".p", ".zip", ".csv")) for name in names)
        manifest = json.load(archive.extractfile(f"{PREFIX}/SOURCE_MANIFEST.json"))
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.extractfile(f"{PREFIX}/{name}").read()).hexdigest() == digest
    second = tmp_path / "second.tar.gz"
    build(second)
    assert output.read_bytes() == second.read_bytes()
