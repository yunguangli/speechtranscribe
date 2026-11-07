import flet as ft

from app import SpeechApp


def main(page: ft.Page) -> None:
    SpeechApp(page)


if __name__ == "__main__":
    ft.run(main)
