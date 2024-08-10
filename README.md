1. Clone the repository to your local system.
2. Open the terminal and navigate to the project directory.
3. Build the Docker image by running the following command:
  ```bash
  docker build -t autotestify .
4. Create docker container using
    docker run -d -p 8888:8888 --name autotestify autotestify
5. Once the server is running, open your preferred browser and navigate to `localhost:8888`.
6. You will be redirected to the homepage of the form.
7. Fill in the required details in the form and click on the 'Submit' button.