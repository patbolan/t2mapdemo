
Running. Activate local environment, then
python3 app.py # debug mode
gunicorn --bind 0.0.0.0:5000 wsgi:app

Before I used to do this:
flask run --host=0.0.0.0

But with the  __init__.py module, you now have to do this:
flask --app . run --host=0.0.0.0

I think the wsgi prefers loading like a module.


I updated the requirements using 
pipreqs . --force
then adding gunicorn

It is not configured for docker as well. Build, run, shytdown, clean up like this (all sudo)
docker build -t my-app .
docker images
docker run -d -p 5000:5000 --name test-container my-app
docker ps
docker stop test-container
docker rm test-container
docker rmi my-app


Notes:
- App listens on 0.0.0.0:5000 by default when run directly.
- Case 105 is the accordian man and rooster.
- Case 001 is the airplane in the medarxiv paper. Fig 2 in Magma and Fig 3 in medarxiv
