import flet
from flet import AppBar, ElevatedButton, Page, Text, View, colors
from flet import Column, Row
from flet import ElevatedButton, FilePicker, FilePickerResultEvent, FilePickerUploadEvent, FilePickerUploadFile
from flet import Page, ProgressRing, Ref, Text, icons


# let's try separating the various stages into separate routes...
# bring in pieces from the image-picker and file-picker tests
# individually, for each route view

# before getting carried away...
# welcome screen (old "go to settings button")
# define arena screen
# define character screen
# define collission screen


def main(page: Page):
    page.title = "GenJinni Alpha"

    # ###########################################################################################
    # file picker stuffs

    prog_bars: Dict[str, ProgressRing] = {}
    files = Ref[Column]()
    upload_button = Ref[ElevatedButton]()

    def file_picker_result(e: FilePickerResultEvent):
        upload_button.current.disabled = True if e.files is None else False
        prog_bars.clear()
        files.current.controls.clear()
        if e.files is not None:
            for f in e.files:
                prog = ProgressRing(value=0, bgcolor="#eeeeee", width=20, height=20)
                prog_bars[f.name] = prog
                files.current.controls.append(Row([prog, Text(f.name)]))
        page.update()

    def on_upload_progress(e: FilePickerUploadEvent):
        prog_bars[e.file_name].value = e.progress
        prog_bars[e.file_name].update()

    file_picker = FilePicker(on_result=file_picker_result, on_upload=on_upload_progress)

    def upload_files(e):
        uf = []
        if file_picker.result is not None and file_picker.result.files is not None:
            for f in file_picker.result.files:
                uf.append(
                    FilePickerUploadFile(
                        f.name,
                        upload_url=page.get_upload_url(f.name, 600),
                    )
                )
            file_picker.upload(uf)

    # hide dialog in a overlay
    page.overlay.append(file_picker)

    # ###########################################################################################
    # Handle the routing and views

    print("Initial route:", page.route)

    def route_change(e):
        print("Route change:", e.route)
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    AppBar(title=Text("Flet app")),
                    ElevatedButton("Build New Game", on_click=open_arena_settings),
                ],
            )
        )
        if page.route == "/arena":
            page.views.append(
                View(
                    "/arena",
                    [
                        AppBar(title=Text("Arena"), bgcolor=colors.SURFACE_VARIANT),
                        Text("Arena Settings!", style="bodyMedium"),
                        ElevatedButton(
                            "Select Arena images...",
                            icon=icons.FOLDER_OPEN,
                            on_click=lambda _: file_picker.pick_files(allow_multiple=True),
                        ),
                        Column(ref=files),
                        ElevatedButton(
                            "Upload Arena",
                            ref=upload_button,
                            icon=icons.UPLOAD,
                            on_click=upload_files,
                            disabled=True,
                        ),
                        ElevatedButton(
                            "Go to character settings", on_click=open_character_settings
                        ),
                    ],
                )
            )
        if page.route == "/character":
            page.views.append(
                View(
                    "/character",
                    [
                        AppBar(title=Text("Character"), bgcolor=colors.SURFACE_VARIANT),
                        Text("Character Settings!", style="bodyMedium"),
                        ElevatedButton(
                            "Select Character images...",
                            icon=icons.FOLDER_OPEN,
                            on_click=lambda _: file_picker.pick_files(allow_multiple=True),
                        ),
                        Column(ref=files),
                        ElevatedButton(
                            "Upload Character",
                            ref=upload_button,
                            icon=icons.UPLOAD,
                            on_click=upload_files,
                            disabled=True,
                        ),
                       ElevatedButton(
                            "Go to collision settings", on_click=open_collision_settings
                        ),
                    ],
                )
            )
        if page.route == "/collision":
            page.views.append(
                View(
                    "/collision",
                    [
                        AppBar(title=Text("Collision"), bgcolor=colors.SURFACE_VARIANT),
                        Text("Collision Settings!", style="bodyMedium"),
                        ElevatedButton(
                            "Select Collision images...",
                            icon=icons.FOLDER_OPEN,
                            on_click=lambda _: file_picker.pick_files(allow_multiple=True),
                        ),
                        Column(ref=files),
                        ElevatedButton(
                            "Upload Collision",
                            ref=upload_button,
                            icon=icons.UPLOAD,
                            on_click=upload_files,
                            disabled=True,
                        ),
                        ElevatedButton(
                            "Go to arena settings", on_click=open_arena_settings
                        ),
                    ],
                )
            )
        if page.route == "/settings" or page.route == "/settings/mail": 
            page.views.append(
                View(
                    "/settings",
                    [
                        AppBar(title=Text("Settings"), bgcolor=colors.SURFACE_VARIANT),
                        Text("Settings!", style="bodyMedium"),
                        ElevatedButton(
                            "Go to mail settings", on_click=open_mail_settings
                        ),
                    ],
                )
            )
        if page.route == "/settings/mail":
            page.views.append(
                View(
                    "/settings/mail",
                    [
                        AppBar(
                            title=Text("Mail Settings"), bgcolor=colors.SURFACE_VARIANT
                        ),
                        Text("Mail settings!"),
                    ],
                )
            )
        page.update()

    def view_pop(e):
        print("View pop:", e.view)
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    def open_mail_settings(e):
        page.go("/settings/mail")

    def open_character_settings(e):
        page.go("/character")

    def open_arena_settings(e):
        page.go("/arena")

    def open_collision_settings(e):
        page.go("/collision")

    def open_settings(e):
        page.go("/settings")

    page.go(page.route)


flet.app(target=main, upload_dir="uploads", view=flet.WEB_BROWSER)