import altair as alt
import seaborn as sns


def main():
    data_penguins = sns.load_dataset("penguins")

    # Define the degree of the polynomial fits
    degree_list = [1, 3, 5]

    base =  alt.Chart(data_penguins).mark_point().encode(
            alt.X("bill_length_mm", scale=alt.Scale(zero=False)),
            alt.Y("bill_depth_mm", scale=alt.Scale(zero=False)),
            color="species",
            tooltip="species",
        )


    polynomial_fit = [
        base.transform_regression(
            "bill_length_mm", "bill_depth_mm", method="poly", order=order, as_=["bill_length_mm", str(order)]
        )
        .mark_line()
        .transform_fold([str(order)], as_=["degree", "bill_depth_mm"])
        .encode(alt.Color("degree:N"))
        for order in degree_list
    ]

    alt.layer(base.interactive(), *polynomial_fit).save("altair_lesson_2.html")

if __name__=='__main__':
    main()
