import polars as pl
import rich


def treating_submission():
    df = pl.read_csv("data/test_tuned_gpt.csv", infer_schema_length=10000)
    print(df.shape)
    df_test = pl.read_csv("data/test.csv")
    print(df_test.shape)

    df = df.with_columns(
        label=pl.when(pl.col("classification") == "1\n")
        .then(1)
        .when(pl.col("classification") == "0, 0, 0, 0, 0, 0, 0, 0, 0, 0")
        .then(0)
        .when(pl.col("classification") == "0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0")
        .then(0)
        .when(pl.col("classification") == "0  \n0")
        .then(0)
        .when(
            pl.col("classification")
            == "Esse tipo de comentário é extremamente tóxico e não deve ser tolerado. Sugerimos que você converse com seu amigo sobre a importância de respeitar a diversidade e promover um ambiente inclusivo. Comentários que desqualificam ou menosprezam alguém não são aceitáveis. É fundamental promover diálogos saudáveis e construtivos, baseados no respeito mútuo e na empatia."
        )
        .then(1)
        .when(pl.col("classification") == "0 ")
        .then(0)
        .when(pl.col("classification") == "0\n0\n0\n0\n0\n0")
        .then(0)
        .when(pl.col("classification") == "0 1 0 0 ")
        .then(0)
        .when(pl.col("classification") == "0\n\n0")
        .then(0)
        .when(
            pl.col("classification")
            == "Peço que evite compartilhar conteúdos desse tipo, pois podem ser prejudiciais e desrespeitosos. Se precisar de ajuda ou orientação sobre algum assunto, estou aqui para ajudar."
        )
        .then(1)
        .when(
            pl.col("classification")
            == "Desculpe, mas não posso ajudar na solicitação de conversas privadas entre pessoas. Posso ajudar com alguma outra questão?"
        )
        .then(1)
        .when(pl.col("classification") == "0 0 0 0")
        .then(0)
        .when(pl.col("classification") == "0")
        .then(0)
        .when(
            pl.col("classification")
            == "Entendi! Se precisar de ajuda com mais alguma coisa, estou por aqui."
        )
        .then(0)
        .when(
            pl.col("classification")
            == "Desculpe, mas não posso participar dessa interação."
        )
        .then(1)
        .when(
            pl.col("classification")
            == "Sinto muito, não posso fornecer opiniões sobre assuntos políticos. A minha função é fornecer informações e auxiliar com dúvidas de forma imparcial. Posso te ajudar com mais alguma coisa?"
        )
        .then(1)
        .when(pl.col("classification") == "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0")
        .then(0)
        .when(pl.col("classification") == "1")
        .then(1)
    )
    df.with_row_index(name="id").select(["id", "label"]).write_csv(
        "data/submission_10.csv"
    )


if __name__ == "__main__":
    treating_submission()
