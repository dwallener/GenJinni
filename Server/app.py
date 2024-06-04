#!/usr/bin/python3

from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.expanduser('~/Sandbox/GenJinn/demo-game-01')
ACTOR_FOLDER = os.path.join(UPLOAD_FOLDER, 'actor')
NPC1_FOLDER = os.path.join(UPLOAD_FOLDER, 'npc1')
NPC2_FOLDER = os.path.join(UPLOAD_FOLDER, 'npc2')
NPC3_FOLDER = os.path.join(UPLOAD_FOLDER, 'npc3')
ARENA_FOLDER = os.path.join(UPLOAD_FOLDER, 'arena')

for folder in [ACTOR_FOLDER, NPC1_FOLDER, NPC2_FOLDER, NPC3_FOLDER, ARENA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    for i in range(4):
        actor_image = request.files.get(f'actor{i}')
        if actor_image:
            actor_image.save(os.path.join(ACTOR_FOLDER, actor_image.filename))
        
        for npc in ['npc1', 'npc2', 'npc3']:
            npc_image = request.files.get(f'{npc}{i}')
            if npc_image:
                npc_image.save(os.path.join(eval(f'{npc.upper()}_FOLDER'), npc_image.filename))
    
    arena_image = request.files.get('arena')
    if arena_image:
        arena_image.save(os.path.join(ARENA_FOLDER, arena_image.filename))

    return 'Files uploaded successfully!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)