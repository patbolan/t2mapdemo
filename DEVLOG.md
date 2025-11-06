** CURRENT WORKING **
Completed compare methods. I was planning on swapping in more examples, but for now, this is just a visualization
So put that off
Complete the invivo page.
Got the fundamentals working. Now format and copy


Need to get the right examples. Looks lkike fig 6 was 400, fig 5 (malignant) was 277



** Running the server**  
Activate local environment, then
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


** Design **
For exploring the datasets
In vivo, the plot should show both raw data and a NLLS fit. 
For synthetic, the plot should show the True data and the synthetic noise.
Noise should be exactly the same as synth

Plan for the data to be retrieved
images are always with noise
predictions
reference 

invivo test data



** Notes ***
- App listens on 0.0.0.0:5000 by default when run directly.
- Case 105 is the accordian man and rooster.
- Case 001 is the airplane in the medarxiv paper. Fig 2 in Magma and Fig 3 in medarxiv

*** TODO ***
* Lots of refactoring and code cleanup!


