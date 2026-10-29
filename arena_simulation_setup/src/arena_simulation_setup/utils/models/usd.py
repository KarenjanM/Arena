import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Collection
from pathlib import Path

import aiofiles

from . import Model, ModelProvider, ModelType


class ModelProvider_USD(ModelProvider.provides(ModelType.USD)):
    @classmethod
    async def load(cls, model_dir: Path, model: str, loader_args: dict | None) -> Model:
        model_paths = (
            model_dir / f"{model}.usdz",
            model_dir / f"{model}.usd",
            model_dir / f"{model}.usda",
            model_dir / f"{model}.usdc",
        )

        found = next(filter(os.path.exists, model_paths), None)
        if found is None:
            raise FileNotFoundError(f"USD model for {model} not found in {model_dir}")
        return Model(
            type=ModelType.USD,
            name=model,
            description="",  # TODO add bytes compat
            path=found,
        )

    @classmethod
    def convertable(cls) -> Collection[ModelType]:
        return (ModelType.SDF,)

    @classmethod
    async def convert(cls, model_dir: Path, model: Model, loader_args: dict | None) -> Model | None:
        if model.type is ModelType.SDF:
            try:
                # print(model_dir)
                model_path = model.path
                assert model_path is not None
                model_dir = model_path.parent
                async with aiofiles.open(model_path) as f:
                    tree = ET.ElementTree(ET.fromstring(await f.read()))
                root = tree.getroot()
                assert root is not None
                # First pass: resolve package:// URIs
                model_uri_pattern = re.compile(r'^model://([^/]+)(.*)$')
                package_uri_pattern = re.compile(r'^package://([^/]+)(.*)$')
                for uri_elem in root.iter():
                    if uri_elem.text:
                        text = uri_elem.text.strip()
                        match = model_uri_pattern.match(text)
                        if match:
                            package_name = match.group(1)
                            remaining_path = match.group(2)
                            # Get the absolute path for the package share directory.
                            # Replace the package URI with the resolved directory plus remaining path.
                            new_uri = model_dir / remaining_path
                            print(new_uri)
                            # uri_elem.text = new_uri
                            if str(new_uri).lower().endswith('.dae') and new_uri.is_file():
                                uri_elem.text = str(await process_dae(new_uri, model_dir))
                            elif str(new_uri).lower().endswith('.obj') and new_uri.is_file():
                                uri_elem.text = str(await process_obj(new_uri, model_dir))
                        else:
                            match = package_uri_pattern.match(text)
                            if match:
                                package_name = match.group(1)
                                remaining_path = match.group(2)
                                # Get the absolute path for the package share directory.
                                # Replace the package URI with the resolved directory plus remaining path.
                                new_uri = model_dir / remaining_path
                                print(new_uri)
                                if str(new_uri).lower().endswith('.dae') and new_uri.is_file():
                                    uri_elem.text = str(await process_dae(new_uri, model_dir))
                                elif str(new_uri).lower().endswith('.obj') and new_uri.is_file():
                                    uri_elem.text = str(await process_obj(new_uri, model_dir))
                model_path = model_dir / model.name / "usd" / f"{model.name}.usd"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if model_path.is_symlink() and not model_path.exists():  # broken symlink
                    model_path.unlink()

                import arena_bringup

                ARENA_DIR = arena_bringup.get_arena_dir()

                env = os.environ.copy()
                env['ARENA_WS_DIR'] = ARENA_DIR
                async with aiofiles.tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
                    ser = ET.tostring(root, encoding="unicode", method="xml")
                    await f.write(ser)
                    await f.flush()
                    assert isinstance(f.name, str)
                    print("Temporary URDF file for converter:", f.name)
                    subprocess.check_output(
                        [f'{ARENA_DIR}/_meta/tools/sdf2usd', f.name, model_path],
                        env=env,
                        # shell=True,
                    )

                from pxr import Usd

                stage = Usd.Stage.Open(model_path)
                for prim in stage.Traverse():
                    if prim.GetTypeName() == "Xform":
                        first_xform_prim = prim
                        break
                else:
                    raise RuntimeError('no xform prim found')
                prim_path = first_xform_prim.GetPath()
                # print(prim_path)
                stage.SetDefaultPrim(first_xform_prim)
                root_layer = stage.GetRootLayer()
                root_layer.Save()
                return await cls.load(model_dir, model.name, loader_args)

            except Exception:
                raise

        return None


async def process_dae(dae_file: Path, package_dir: Path) -> Path:
    """
    Load a .dae file, update its <init_from> elements by replacing any leading
    '../' with the package_dir, then write to a temporary file and return its path.
    """
    import collada

    file = collada.Collada(str(dae_file))
    tree = file.xmlnode
    root = tree.getroot()
    assert root is not None
    for init_elem in root.iterfind('.//init_from'):
        print(init_elem)
        if init_elem.text:
            text = init_elem.text.strip()
            if text.startswith("../"):
                # Remove all leading "../" segments
                rel_path = text
                while rel_path.startswith("../"):
                    rel_path = rel_path[3:]
                # Create a new absolute path using the package directory
                new_text = os.path.join(package_dir, rel_path)
                init_elem.text = new_text

    async with aiofiles.tempfile.NamedTemporaryFile(mode="wb", suffix=".dae", delete=False) as tmp_file:
        # Write the XML tree to the temporary file.
        ser = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True)
        await tmp_file.write(ser)
    assert isinstance(tmp_file.name, str)
    return Path(tmp_file.name)

    # Write the updated .dae file to a temporary file


async def process_obj(obj_file: Path, package_dir: Path) -> Path:
    """
    Read an .obj file as text and update any .png file references.
    For any found relative .png path (e.g. starting with "../"), remove the
    relative segments and prepend the package_dir. The modified file is saved
    to a temporary file whose path is returned.
    """
    try:
        async with aiofiles.open(obj_file, encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        print(f"Error reading {obj_file}: {e}")
        return obj_file  # fallback: return original file if error occurs

    # Regex to match .png filenames (non-space characters ending in .png)
    png_pattern = re.compile(r'(?P<path>\S+\.png)')
    mtl_patter = re.compile(r'(?P<path>\S+\.mtl)')

    def replace_png(match: re.Match[str]) -> str:
        path = match.group("path")
        # If already absolute, do nothing.
        if os.path.isabs(path):
            return path
        # Remove any leading '../' segments
        while path.startswith("../"):
            path = path[3:]
        # Return the absolute path by joining with the package directory
        return os.path.join(package_dir, path)

    new_content = png_pattern.sub(replace_png, content)

    # Write the updated .obj file to a temporary file
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix='.obj', mode='w', encoding='utf-8') as temp_file:
        await temp_file.write(new_content)
    assert isinstance(temp_file.name, str)
    return Path(temp_file.name)
