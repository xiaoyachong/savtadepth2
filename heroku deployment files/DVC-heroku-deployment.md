We need to give Heroku the ability to pull in data from DVC upon app start up. We will install a [buildpack](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt) that allows the installation of apt-files and then define the Aptfile that contains a path to DVC. I.e., in the CLI run:

```
heroku buildpacks:add --index 1 heroku-community/apt
```

Then in your root project folder create a file called `Aptfile` that specifies the release of DVC you want installed, https://github.com/iterative/dvc/releases/download/2.8.3/dvc_2.8.3_amd64.deb

Add the following code block to your **streamlit_app.py**:

```python
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system(f"dvc pull {model} {image}") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
```

Reference: [Heroku ML](https://github.com/GuilhermeBrejeiro/deploy_ML_model_Heroku_FastAPI)