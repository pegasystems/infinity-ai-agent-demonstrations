import reflex as rx

config = rx.Config(
    app_name="DeepEval_Pega",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                accent_color="blue",
                gray_color="slate",
                radius="medium",
            ),
        ),
    ],
    # Pin react-syntax-highlighter to 15.6.1; v16.1.0 ships with a broken ESM build
    frontend_packages=["react-syntax-highlighter@15.6.1"],
)