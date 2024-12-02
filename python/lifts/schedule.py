import pandas as pd
import polars as pl
import lifts.utilities
import lifts.dictionary

def build_train_timetable(train_consist_plan, terminal, swap_arrive_depart, as_dicts):
    df = (train_consist_plan
        .filter(
            pl.col("Destination_ID") == pl.lit(terminal), 
            pl.col("Train_Type").str.starts_with(pl.lit("Intermodal"))
        )
        .rename({
            "Train_ID": "train_id",
            "Departure_Time_Actual_Hr": "departure_time",
            "Arrival_Time_Actual_Hr": "arrival_time",
            "Cars_Empty": "empty_cars",
            "Cars_Loaded": "full_cars"
        })
    ) 

    if swap_arrive_depart:
        df = df.rename({"departure_time": "arrival_time", "arrival_time": "departure_time"})

    df = (df
        .group_by("train_id")
            .agg(pl.col("full_cars", "empty_cars", "arrival_time", "departure_time").first())
        .sort("arrival_time", descending=False)
    )

    if as_dicts:
        return (df
            .with_columns(
                pl.lit(lifts.dictionary.calculate_oc_number()).alias("oc_number"),
            )
            .pipe(lifts.dictionary.calculate_truck_number)
            .to_dicts()
        )
    else:
        return df.to_pandas()


def next_train_timetable(train_id, terminal):
    df_terminal = build_train_timetable(terminal, swap_arrive_depart = False, as_dicts = False)
    df_next_train = df_terminal.iloc[train_id]
    return df_next_train


def outbound_containers():
    df = pd.read_csv('C:/Users/Irena Tong/PycharmProjects/simulation_test/data/outbound_plan.csv')
    return df


def get_next_train_outbound_data(index):
    outbound_df = outbound_containers()
    outbound_num = outbound_df.iloc[index]['Outbound_Num']
    return outbound_num

# # Test codes
# terminal = 'Allouez'
# print(train_timetable(terminal))
#
# next_train = next_train_timetable(1, terminal)
# print(next_train)
#
# next_outbound_num = get_next_train_outbound_data(1)
# print(next_outbound_num)