import os
import fire
import json
import requests
import getpass

from pathlib import Path
from typing import Dict


class CLI:
    def signup():
        """
        Sign up to the Daisytuner backend.
        """
        print("Welcome to the Daisytuner SDK.")
        print()

        print("Sign up with e-mail and password.")
        email = input("E-Mail: ")

        password = None
        for _ in range(3):
            psw = getpass.getpass("Password: ")
            psw2 = getpass.getpass("Repeat: ")
            if psw == psw2:
                password = psw
                break
            else:
                print("Passwords did not match. Try again.")

        if psw is None:
            print("Sign up failed.")
            exit(0)

        req = requests.post(
            "https://signup-bhqsvyw3sa-uc.a.run.app",
            json={
                "email": email,
                "password": password,
            },
        )
        if not req.ok:
            print("Sign up failed: ", req.content)
            exit(0)

        print("Success! You're now logged in.")
        CLI._save_user(req.json())

    def login(email: str = None, password: str = None):
        """
        Log in to the Daisytuner backend.
        """
        print("Welcome to the Daisytuner SDK.")
        print()

        print("Log in with e-mail and password.")
        if email is None:
            email = input("E-Mail: ")
        if password is None:
            password = getpass.getpass("Password: ")

        req = requests.post(
            "https://login-bhqsvyw3sa-uc.a.run.app",
            json={
                "email": email,
                "password": password,
            },
        )
        if not req.ok:
            print("Log in failed: ", req.content)
            exit(0)

        print("Success! You're now logged in.")
        CLI._save_user(req.json())

    def logout():
        """
        Log out to the Daisytuner backend.
        """
        CLI._remove_user()

    @staticmethod
    def user():
        current_user = CLI._load_user()

        # # Refresh token
        # req = requests.post(
        #     "https://securetoken.googleapis.com/v1/token?key=*",
        #     json={
        #         "grant_type": "refresh_token",
        #         "refresh_token": current_user["refreshToken"],
        #     },
        # )
        # if not req.ok:
        #     raise ValueError("Authentication: Refresh token failed")

        # payload = req.json()
        # current_user["refreshToken"] = payload["refresh_token"]
        # current_user["idToken"] = payload["id_token"]
        # CLI._save_user(current_user)

        return current_user

    @staticmethod
    def _save_user(payload: Dict):
        daisy_path = Path.home() / ".daisytuner"
        daisy_path.mkdir(exist_ok=True, parents=False)
        with open(daisy_path / "token.json", "w") as handle:
            json.dump(payload, handle)

    @staticmethod
    def _remove_user():
        daisy_path = Path.home() / ".daisytuner"
        if (daisy_path / "token.json").is_file():
            os.remove(daisy_path / "token.json")

    @staticmethod
    def _load_user():
        daisy_path = Path.home() / ".daisytuner"
        daisy_path.mkdir(exist_ok=True, parents=False)
        if not (daisy_path / "token.json").is_file():
            raise ValueError(
                "Authentication: No user found. Try running `daisytuner login`"
            )

        with open(daisy_path / "token.json", "r") as handle:
            user = json.load(handle)
            return user


def main():
    fire.Fire({"signup": CLI.signup, "login": CLI.login, "logout": CLI.logout})
