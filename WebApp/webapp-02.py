import os

from nicegui import ui
#from nicegui.events import ValueChangeEventArguments
from nicegui import events, ui
from nicegui import Tailwind

# globals
upload_dir = f'uploads'
level_dir = f'{upload_dir}/level'
actor_dir = f'{upload_dir}/actor'
npc_dir = f'{upload_dir}/npc'
physics_dir = f'{upload_dir}/physics'

#
# Helper functions
# 


def show(event: events.ValueChangeEventArguments):
    name = type(event.sender).__name__
    ui.notify(f'{name}: {event.value}')


def handle_upload(e: events.UploadEventArguments, section, idx):

    b = e.content.read()

    if section == 'actor':
        outdir = actor_dir

    if section == 'npc':
        outdir = npc_dir

    if section == 'physics':
        outdir = physics_dir

    if section == 'level':
        outdir = level_dir

    with open(os.path.join(outdir, f'{section}-{idx}.png'), "wb") as file:
        file.write(b)

    ui.notify(f'Image saved to {outdir}')


with ui.button():
    ui.label('Welcome! Jinnie is here to help.')
    ui.image('images/genjinn-large.png')

#
# Snippets for later
#

# for capturing text input
# input_box_of_interest = ui.input() input_box_of_interest.value



#
# level stuff
#

ui.input(label='Level Name:', placeholder='start typing',
         on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.label('Upload the level artwork')
ui.upload(on_upload=lambda e: handle_upload(e, 'level', 0), 
          on_rejected=lambda e: ui.notify(f'Uploaded...')).classes('max-w-full').props('accept=.png')

#
# actor stuff
#

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the main character, or actor...').tailwind.font_weight('extrabold')

with ui.grid(columns=3):

    ui.label('Upload the main character (ie front view)')
    ui.label('Upload the main character (ie side view)')
    ui.label('Upload the main character (ie back view)')

    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 0), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 1), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 2), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.label('Add movement keyframes')
    ui.label('Add movement keyframes')
    ui.label('Add movement keyframes')

    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 3), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 4), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'actor', 5), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.button('Add another animation')

#
# npc stuff
#

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the NPCs...')

with ui.grid(columns=4):

    ui.label('Upload the NPC 1 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 0), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 1), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 2), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 3), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.label('Upload the NPC 2 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 4), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 5), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 6), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 7), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.label('Upload the NPC 3 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 8), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 9), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 10), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'npc', 11), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.button('Add another NPC')

#
# physics stuff
#

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the physics and collisions...')

with ui.grid(columns=4):

    ui.label('Upload what collision 1 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 0), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 1), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 2), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 3), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.label('Upload what collision 2 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 4), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 5), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 6), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 7), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.label('Upload what collision 3 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 8), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 9), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 10), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')
    ui.upload(on_upload=lambda e: handle_upload(e, 'physics', 11), 
            on_rejected=lambda e: ui.notify(f'Uploaded...{e.name}').classes('max-w-full'),
            label='level').classes('max-w-full').props('accept=.png')

    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.label('Tell Jinnie about this image...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.button('Add another Collion/Physics')

# submit

with ui.button():
    ui.label('Submit & Generate!')
    ui.image('images/genjinn-large.png')

ui.run()

