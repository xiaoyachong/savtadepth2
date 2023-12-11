import torch
import os
from fastai.vision.all import *
import gradio as gr

############### HF ###########################

HF_TOKEN = os.getenv("HF_TOKEN")

hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "savtadepth-flags")

############## DVC ################################

PROD_MODEL_PATH = "src/models"
TRAIN_PATH = "src/data/processed/train/bathroom"
TEST_PATH = "src/data/processed/test/bathroom"

if os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config cache.type copy")
    os.system("dvc config core.no_scm true")
    if os.system(f"dvc pull {PROD_MODEL_PATH} {TRAIN_PATH } {TEST_PATH }") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc")
# .apt/usr/lib/dvc

############## Inference ##############################


class ImageImageDataLoaders(DataLoaders):
    """Basic wrapper around several `DataLoader`s with factory methods for Image to Image problems"""

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(
        cls,
        path,
        filenames,
        label_func,
        valid_pct=0.2,
        seed=None,
        item_transforms=None,
        batch_transforms=None,
        **kwargs,
    ):
        """Create from list of `fnames` in `path`s with `label_func`."""
        datablock = DataBlock(
            blocks=(ImageBlock(cls=PILImage), ImageBlock(cls=PILImageBW)),
            get_y=label_func,
            splitter=RandomSplitter(valid_pct, seed=seed),
            item_tfms=item_transforms,
            batch_tfms=batch_transforms,
        )
        res = cls.from_dblock(datablock, filenames, path=path, **kwargs)
        return res


def get_y_fn(x):
    y = str(x.absolute()).replace(".jpg", "_depth.png")
    y = Path(y)

    return y


def create_data(data_path):
    fnames = get_files(data_path / "train", extensions=".jpg")
    data = ImageImageDataLoaders.from_label_func(
        data_path / "train",
        seed=42,
        bs=4,
        num_workers=0,
        filenames=fnames,
        label_func=get_y_fn,
    )
    return data


data = create_data(Path("src/data/processed"))
learner = unet_learner(
    data, resnet34, metrics=rmse, wd=1e-2, n_out=3, loss_func=MSELossFlat(), path="src/"
)
learner.load("model")


def gen(input_img):
    return PILImageBW.create((learner.predict(input_img))[0]).convert("L")


################### Gradio Web APP ################################

title = "SavtaDepth WebApp"

description = """
<p>
<center>
Savta Depth is a collaborative Open Source Data Science project for monocular depth estimation - Turn 2d photos into 3d photos. To test the model and code please check out the link bellow.
<img src="https://huggingface.co/spaces/kingabzpro/savtadepth/resolve/main/examples/cover.png" alt="logo" width="250"/>
</center>
</p>
"""

article = "<p style='text-align: center'><a href='https://dagshub.com/OperationSavta/SavtaDepth' target='_blank'>SavtaDepth Project from OperationSavta</a></p><p style='text-align: center'><a href='https://colab.research.google.com/drive/1XU4DgQ217_hUMU1dllppeQNw3pTRlHy1?usp=sharing' target='_blank'>Google Colab Demo</a></p></center></p>"

examples = [
    ["examples/00008.jpg"],
    ["examples/00045.jpg"],
]
favicon = "examples/favicon.ico"
thumbnail = "examples/SavtaDepth.png"


def main():
    iface = gr.Interface(
        gen,
        gr.inputs.Image(shape=(640, 480), type="numpy"),
        "image",
        title=title,
        flagging_options=["incorrect", "worst", "ambiguous"],
        allow_flagging="manual",
        flagging_callback=hf_writer,
        description=description,
        article=article,
        examples=examples,
        theme="peach",
        allow_screenshot=True,
    )

    iface.launch(enable_queue=True)


# enable_queue=True,auth=("admin", "pass1234")

if __name__ == "__main__":
    main()

