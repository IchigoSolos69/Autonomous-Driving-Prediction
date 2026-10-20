"""Let the user approve read-only notebook access through the official SDK."""

def main():
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_oauth import KaggleOAuth

    print('Approve read-only notebook access in your browser. Never paste credentials into chat.')
    with KaggleClient() as client:
        KaggleOAuth(client).authenticate(scopes=['kernels.get:*'])
    print('Kaggle connection saved locally by the official SDK.')


if __name__ == '__main__':
    main()
