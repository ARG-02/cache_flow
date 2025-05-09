import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

output_dir = "images/100_images"
os.makedirs(output_dir, exist_ok=True)

# Define callback function that includes prompt index
def make_callback(prompt_idx):
    def save_intermediate(step: int, timestep: int, latents):
        with torch.no_grad():
            image = pipe.vae.decode(latents / 0.18215).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            img = Image.fromarray((image * 255).astype("uint8"))
            filename = f"{output_dir}/prompt_{prompt_idx:03d}_step_{step:03d}.png"
            img.save(filename)
    return save_intermediate

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

animal_sports_prompts = [
    "a cartoon lion kicking a soccer ball in a colorful stadium",
    "a cartoon panda shooting a basketball with a big grin",
    "a cartoon dolphin spiking a volleyball underwater with bubbles",
    "a cartoon raccoon playing street hockey in bright city streets",
    "a cartoon giraffe dunking a basketball with exaggerated motion",
    "a cartoon tiger sprinting on a rainbow-colored track field",
    "a cartoon polar bear playing ice hockey with a tiny helmet",
    "a cartoon kangaroo boxing with oversized gloves in a ring",
    "a cartoon cheetah racing with flames coming from its sneakers",
    "a cartoon monkey hitting a home run in a jungle baseball field",
    "a cartoon elephant playing rugby and charging with the ball",
    "a cartoon fox playing ping pong with rapid-fire moves",
    "a cartoon squirrel balancing on a gymnastics beam mid-flip",
    "a cartoon rhino smashing through defenders in a football game",
    "a cartoon chimpanzee saving a soccer goal with a dramatic dive",
    "a cartoon penguin sliding into home base with a baseball helmet",
    "a cartoon flamingo serving a tennis ball on a pastel court",
    "a cartoon koala playing cricket with a eucalyptus bat",
    "a cartoon husky doing snowboarding tricks in the clouds",
    "a cartoon crocodile catching a water polo ball with goggles",
    "a cartoon parrot skydiving with teammates in rainbow suits",
    "a cartoon hippo doing synchronized swimming in sparkly water",
    "a cartoon rabbit riding a motocross bike over dirt hills",
    "a cartoon bear lifting huge barbells with veins popping",
    "a cartoon meerkat leaping to catch a frisbee mid-air",
    "a cartoon goat pole vaulting over a cotton candy stadium",
    "a cartoon cat rock climbing with a determined face",
    "a cartoon llama spiking a volleyball at the beach",
    "a cartoon owl aiming a bow and arrow at a bullseye",
    "a cartoon snake bowling a perfect strike, tongue out",
    "a cartoon dog and cat facing off in a chess championship",
    "a cartoon seal riding a wave on a surfboard with sunglasses",
    "a cartoon deer cradling a lacrosse ball in a forest stadium",
    "a cartoon sloth running a marathon very slowly, cheering crowd",
    "a cartoon horse dribbling a basketball on a farm hoop",
    "a cartoon toucan leaping across rooftops doing parkour",
    "a cartoon squirrel skating a half-pipe with cool tricks",
    "a cartoon rooster fencing with dramatic flair and feathers flying",
    "a cartoon frog leaping over track hurdles with bug-eyed focus",
    "a cartoon moose sweeping the ice in a curling match",
    "a cartoon beaver doing BMX flips with a helmet and shades",
    "a cartoon goldfish diving off a board into a fish tank",
    "a cartoon camel kicking a soccer ball in the desert",
    "a cartoon leopard skiing down a snowy slope super fast",
    "a cartoon pelican catching a frisbee mid-flight",
    "a cartoon duck wrestling a chicken in a wrestling ring",
    "a cartoon snake doing yoga poses on a peaceful mat",
    "a cartoon parrot running a relay race with a baton in beak",
    "a cartoon turtle on a skateboard zooming down a ramp",
    "a cartoon buffalo slam-dunking a basketball",
    "a cartoon dolphin twirling in an ice skating routine",
    "a cartoon porcupine breaking boards in a karate dojo",
    "a cartoon lizard doing a dance move on a sports stage",
    "a cartoon monkey swinging through jungle gymnastic rings",
    "a cartoon zebra and horse playing doubles tennis",
    "a cartoon cat swinging a golf club on a mini-golf course",
    "a cartoon panda aiming an arrow at a target in a forest",
    "a cartoon falcon racing drones through the sky",
    "a cartoon goat leaping over a high jump bar",
    "a cartoon crab playing beach volleyball with big claws",
    "a cartoon giraffe shooting free throws in a school gym",
    "a cartoon duck doing backflips in a diving competition",
    "a cartoon hamster dribbling a basketball across a hamster wheel",
    "a cartoon fox doing parkour flips between tree branches",
    "a cartoon raccoon playing tennis with glowing rackets",
    "a cartoon elephant doing shot put with a giant peanut",
    "a cartoon llama playing dodgeball in a gymnasium",
    "a cartoon octopus playing ping pong with all 8 arms",
    "a cartoon squirrel skiing on marshmallow snow hills",
    "a cartoon lion breakdancing at a halftime show",
    "a cartoon koala doing fencing on a mountaintop arena",
    "a cartoon moose running hurdles with giant antlers",
    "a cartoon bear jumping on a trampoline with flips",
    "a cartoon dog skating a ramp with paw print designs",
    "a cartoon sloth doing slow-motion weightlifting",
    "a cartoon pig doing gymnastics flips in a pink leotard",
    "a cartoon turtle doing martial arts in slow motion",
    "a cartoon rabbit competing in archery at the carrot olympics",
    "a cartoon pelican doing BMX tricks in the clouds",
    "a cartoon cat on rollerblades doing cool dance moves",
    "a cartoon panda playing badminton with bamboo rackets",
    "a cartoon cheetah doing hurdles with lightning bolts",
    "a cartoon frog on a pogo stick at a jungle fair",
    "a cartoon parrot doing gymnastics on uneven bars",
    "a cartoon squirrel racing through a cheese maze course",
    "a cartoon owl doing synchronized diving under moonlight",
    "a cartoon deer spinning a basketball on its antlers",
    "a cartoon monkey bouncing on a trampoline with bananas",
    "a cartoon walrus playing curling with fish-shaped stones",
    "a cartoon raccoon riding a unicycle in a bike race",
    "a cartoon duck sprinting with exaggerated flappy wings",
    "a cartoon giraffe serving a tennis ball with a tall racket",
    "a cartoon tiger snowboarding down a rainbow ramp",
    "a cartoon hippo breakdancing in a hip-hop battle",
    "a cartoon puppy and kitten doing synchronized swimming",
    "a cartoon mouse doing Olympic-level gymnastics vaults",
    "a cartoon goat leaping into a basketball dunk pose",
    "a cartoon eagle hang gliding in a mountain sports fest",
    "a cartoon seal riding a scooter in a skate park",
    "a cartoon parrot water skiing behind a speedboat",
    "a cartoon rabbit doing archery blindfolded, bullseye hit",
    "a cartoon fox running through a jungle obstacle course",
    "a cartoon turtle doing a handstand at a yoga retreat"
]

for idx, prompt in enumerate(animal_sports_prompts):
    print(f"Generating image {idx + 1}/100: {prompt}")
    image = pipe(prompt, callback=make_callback(idx), callback_steps=1).images[0]
    final_path = f"{output_dir}/prompt_{prompt_idx:03d}_step_50.png"
    image.save(final_path)

