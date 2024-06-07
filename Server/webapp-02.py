from nicegui import ui
from nicegui.events import ValueChangeEventArguments

def show(event: ValueChangeEventArguments):
    name = type(event.sender).__name__
    ui.notify(f'{name}: {event.value}')

ui.button('Welcome! This is the magic help button!', on_click=lambda: ui.notify('Click'))

# level stuff

ui.input(label='Level Name:', placeholder='start typing',
         on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.label('Upload the level artwork')
# TODO: how do I save this file?
ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

# actor stuff

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the main character, or actor...')

with ui.grid(columns=3):
    ui.label('Upload the main character')
    ui.label('Upload the main character')
    ui.label('Upload the main character')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.label('Add a description...')
    ui.label('Add a description...')
    ui.label('Add a description...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))
    ui.label('Add movement keyframes')
    ui.label('Add movement keyframes')
    ui.label('Add movement keyframes')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.label('Add a description...')
    ui.label('Add a description...')
    ui.label('Add a description...')
    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

    ui.textarea(label='Describe', placeholder='start typing',
                on_change=lambda e: result.set_text('you typed: ' + e.value))

ui.button('Add another animation')

# npc stuff

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the NPCs...')

with ui.grid(columns=4):

    ui.label('Upload the NPC 1 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

    ui.label('Upload the NPC 2 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

    ui.label('Upload the NPC 3 character')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

ui.button('Add another NPC')

# physics stuff

# TODO: Highlight/bold this line, maybe add separator
ui.label('Desribe the physics and collisions...')

with ui.grid(columns=4):

    ui.label('Upload what collision 1 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

    ui.label('Upload what collision 2 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

    ui.label('Upload what collision 3 looks like')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')
    ui.label('Animation keyframe ')

    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

ui.button('Add another NPC')

# submit

ui.button('SUBMIT')

ui.run()

