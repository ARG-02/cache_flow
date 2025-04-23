import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, OrderedDict

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler

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

animal_sports_prompts2 = [
    "a cartoon owl playing basketball with a tiny ball",
    "a cartoon bear climbing a rock wall in a sports competition",
    "a cartoon elephant ice skating on a frozen pond",
    "a cartoon giraffe running the 100m sprint with long strides",
    "a cartoon fox jumping through flaming hoops in an obstacle course",
    "a cartoon kangaroo playing volleyball with friends",
    "a cartoon tiger doing a backflip in a gymnastics routine",
    "a cartoon rabbit throwing a javelin in a meadow",
    "a cartoon sloth doing yoga poses in a tranquil park",
    "a cartoon monkey playing tennis against a giant raccoon",
    "a cartoon panda surfing big waves on a bamboo board",
    "a cartoon hippo playing rugby with a giant rugby ball",
    "a cartoon owl swooping down to catch a frisbee",
    "a cartoon zebra doing pole dancing at a circus event",
    "a cartoon deer doing a triple jump in a field",
    "a cartoon lion doing sumo wrestling in a match",
    "a cartoon raccoon swimming through an underwater obstacle course",
    "a cartoon giraffe doing a cheerleading jump in a stadium",
    "a cartoon turtle racing in a speedboat on a lake",
    "a cartoon squirrel doing parkour on a jungle gym",
    "a cartoon owl competing in a high dive competition",
    "a cartoon sloth curling in a winter sports competition",
    "a cartoon cheetah ice climbing up a steep mountain",
    "a cartoon panda participating in a triathlon",
    "a cartoon lion bungee jumping with a parachute",
    "a cartoon rabbit playing croquet with giant mallets",
    "a cartoon tiger running the hurdles in a track meet",
    "a cartoon wolf playing American football in a stadium",
    "a cartoon gorilla competing in a weightlifting contest",
    "a cartoon frog kayaking down a river at top speed",
    "a cartoon monkey competing in an obstacle course race",
    "a cartoon beaver making a basketball dunk with a massive hoop",
    "a cartoon elephant jumping rope with a giant rope",
    "a cartoon koala doing snowball fights on a winter day",
    "a cartoon flamingo breakdancing with a crew",
    "a cartoon parrot kayaking through whitewater rapids",
    "a cartoon horse competing in barrel racing",
    "a cartoon lion playing tennis with a giant ball",
    "a cartoon cheetah skateboarding through the city",
    "a cartoon sloth participating in a hula hoop contest",
    "a cartoon raccoon playing rugby with a football-shaped frisbee",
    "a cartoon octopus playing football with all eight tentacles",
    "a cartoon zebra ice skating on a frozen lake",
    "a cartoon giraffe rock climbing on a huge mountain",
    "a cartoon cheetah sprinting through a field of flowers",
    "a cartoon turtle running a marathon at an agonizing pace",
    "a cartoon owl shooting basketballs through an enormous hoop",
    "a cartoon monkey swing dancing at a competition",
    "a cartoon penguin racing in a sledding competition",
    "a cartoon lion doing parkour across city rooftops",
    "a cartoon koala shooting hoops in a basketball gym",
    "a cartoon rabbit playing badminton with oversized rackets",
    "a cartoon toucan surfing on a massive wave",
    "a cartoon snake doing a crossfit workout",
    "a cartoon cat doing a perfect dive off a high board",
    "a cartoon fox participating in a synchronized swimming competition",
    "a cartoon panda running a 10k race in a bamboo forest",
    "a cartoon wolf skiing down a steep mountain with style"
]

class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.min_freq = 0
        self.cache = {}
        self.freqs = defaultdict(OrderedDict)

    def get(self, key):
        if key not in self.cache:
            return None
        value, freq = self.cache[key]

        del self.freqs[freq][key]
        if not self.freqs[freq]:
            del self.freqs[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        self.cache[key] = (value, freq + 1)
        self.freqs[freq + 1][key] = None
        return value

    def put(self, key, value):
        if self.capacity == 0:
            return

        if key in self.cache:
            _, freq = self.cache[key]
            self.cache[key] = (value, freq)
            self.get(key)
            return

        if len(self.cache) >= self.capacity:
            evict_key, _ = self.freqs[self.min_freq].popitem(last=False)
            del self.cache[evict_key]
            if not self.freqs[self.min_freq]:
                del self.freqs[self.min_freq]

        self.cache[key] = (value, 1)
        self.freqs[1][key] = None
        self.min_freq = 1

def init_cache(dir_path):
    cache = LFUCache(capacity=len(animal_sports_prompts))
    idx = 0
    count = 0

    filenames = sorted(os.listdir(dir_path))
    while count < len(animal_sports_prompts):
        intermediates = []
        for i in range(50):
            intermediates.append(dir_path + "/" + filenames[idx * 51 + i])

        cache.put(animal_sports_prompts[count], intermediates)
        count += 1
        idx += 1

    return cache

def make_callback(prompt_idx):
    def save_intermediate(step: int, timestep: int, latents):
        with torch.no_grad():
            image = pipe.vae.decode(latents / 0.18215).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            img = Image.fromarray((image * 255).astype("uint8"))
            filename = f"{temp_dir}/prompt_{prompt_idx:03d}_step_{step:03d}.png"
            img.save(filename)
    return save_intermediate

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    input_dir = "images/100_images"
    if not os.path.exists(input_dir):
        print("Images not found!!")
        exit

    output_dir = "comprison/50_images"
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = "temp/50_images"
    os.makedirs(temp_dir, exist_ok=True)

    cache = init_cache(input_dir)

    model_id = "stabilityai/stable-diffusion-2-1"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe2 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe2.scheduler = DPMSolverMultistepScheduler.from_config(pipe2.scheduler.config)
    pipe2 = pipe2.to("cuda")

    prefix = ""
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    model = model.to(device)

    max_length = 32768
    k = 50
    strength = 0.2

    for idx, prompt in enumerate(animal_sports_prompts2):
        print(f"Generating image {idx + 1}/50: {prompt}")

        p = list(cache.cache.keys())

        embs = model.encode(p, instruction=prefix, max_length=max_length)
        prompt_emb = model.encode([prompt], instruction=prefix, max_length=max_length)
    
        embs = F.normalize(embs, p=2, dim=1)
        prompt_emb = F.normalize(prompt_emb, p=2, dim=1)

        cosine_sim = (embs@prompt_emb.T).squeeze()
        max_value, max_index = torch.max(cosine_sim, dim=0)

        steps = int((20 / 0.6) * max_value)

        print(f"Max cosine similarity is {max_value}, starting at {steps}, taking {k-steps} steps. ")

        states = cache.get(p[max_index])
        init_image = Image.open(states[steps]).convert("RGB")
        result = pipe(prompt=prompt, image=init_image, strength=strength, num_inference_steps=(k-steps))
        final_path = f"{output_dir}/prompt_{idx:03d}_pre.png"
        output_image = result.images[0]
        output_image.save(final_path)
 
        image = pipe2(prompt, callback=make_callback(idx), callback_steps=1).images[0]
        final_path = f"{output_dir}/prompt_{idx:03d}_noise.png"
        image.save(final_path)

        filenames = sorted(os.listdir(temp_dir))
        intermediates = []
        for i in range(50):
            intermediates.append(temp_dir + "/" + filenames[idx * 50 + i])

        cache.put(animal_sports_prompts2[idx], intermediates)
