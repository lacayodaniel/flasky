# Traffic Sign Recognition DNN

## Set-up and Installation
This project utilizes `pipenv`, the package manager. Open a terminal in the project directory and..

1. Install pipenv
`pip install pipenv`
2. Start a virtual environment specific to this project
`pipenv shell`
3. Install all packages from pipfile to the virtual environment
`pipenv install`


Type `FLASK_APP=hello.py FLASK_ENV=development flask run` and navigate to 127.0.0.1 in your browser.


## Pages
### /user/{name}
A dynamic user layout scheme is in place. Navigate to `/user/Dog` and `/user/Cat` to see the difference.

### /graph
This page generates an example matplotlib figure, saves the data as an image png, and uses html <img> tags to render the graph.

### /test1
This page rotates a traffic sign 90 degrees, classifying the sign for each degree of rotation. The accuracy is computed and the corresponding rotation angle and classification is visualized.

![ScreenShot](/screenshots/test1.png)
