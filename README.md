<img src="frontend/images/Seatbelt.png" alt="image" width="300">

## Inspiration
According to the Institute of Global Politics, a study in 2023 by US cybersecurity firm Home Security Heroes reported that **deepfake porn makes up 98 percent of all deepfake videos** online, with **99 percent of them targeting women.**

This practice is becoming **"increasingly prevalent among middle and high school students."** At a high school in Washington, a "male student was caught making sexually explicit deepfake of 14 and 15 year-old female classmates."

We find this **horrifying** and completely unacceptable. 

Instead of ignoring this **entirely legal practice,** we decided to build an image editing app that adds an ai-prevention filter to protect you from being made part of nonconsensual explicit material.

## What it does
**Seatbelt is here to keep you safe when driving online!**

Seatbelt is a photo editing application that applies a filter to your image, thus perturbing it and making it difficult for ai to perform deepfakes. 

## How we built it
### Front End
We used raw HTML, CSS, and JavaScript to built our front end to avoid bloat, creating a lightning-fast user experience. The front end hooks up to the backend, which uses genai to perturb the images and perform face swaps.

### Back End
We used a state-of-the-art technique from the scientific literature called unGANable (paper linked below) to add a layer of noise onto images to make ai tools unable to understand what's going on in the image. We wrote the implementation from scratch using onnx and pytorch for the resnet 50 model that we employ for feature extraction. We then use a gradient-based perturbation method from the paper to minimize distance from the image while maximizing the distance in the latent space.

https://www.usenix.org/conference/usenixsecurity23/presentation/li-zheng

### Deployment
Thanks to MLH we registered our domain name and used a Lambda Labs instance to get access to GPU compute for inferencing on a Tesla V100-SXM2-16GB. Therefore we have a fully deployed and efficiently resourced web app for AI inferencing and backend server access.

## Challenges we ran into
Acquiring enough GPU, literature that isn't advanced enough for modern face-swapping techniques.

## Accomplishments that we're proud of
We built a fully-deployed, fully-functioning webapp.
We didn't use any JavaScript frameworks, instead choosing to use raw HTML. 

Also we found a bug in PyTorch!
Also also we drew the cat logo by hand, no AI involved!

## What we learned
We knew that deepfake pornography was an existing problem, but we didn't know the prevalence until researching the stats.

We also learned that it is depressingly difficult to filter the image enough to defeat modern face-swapping technology. Even the paper that we were working off of didn't work out of the box.

## What's next for SeatBelt
We would love to integrate this filter into the cameras app, allowing people around the world to protect themselves online.
