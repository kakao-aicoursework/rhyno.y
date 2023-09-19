import pynecone as pc

class MarketingexConfig(pc.Config):
    pass

config = MarketingexConfig(
    app_name="marketingEx",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)