# import packages
import warnings

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

warnings.filterwarnings("ignore")

results = defaultdict(list)
game_preds = defaultdict(list)
linr_preds = defaultdict(list)

for TEST_YEAR in range(2005, 2023):

    metrics = ['point diff', 'point ratio', 'pythagorean point ratio', 'dsr', 'game control', 'game control signed',
               'game control avg', 'game control fourth']
    calibration_years = list(range(TEST_YEAR - 7, TEST_YEAR + 1)) \
        if TEST_YEAR > 2020 \
        else list(range(TEST_YEAR - 6, TEST_YEAR + 1))

    if 2020 in calibration_years and calibration_years[-1] != 2020:
        calibration_years.remove(2020)

    games = nfl.import_schedules(calibration_years)
    games.dropna(axis=0)
    games['home_team'].replace({'SD': 'LAC', 'OAK': 'LV', 'STL': 'LA'}, inplace=True)
    games['away_team'].replace({'SD': 'LAC', 'OAK': 'LV', 'STL': 'LA'}, inplace=True)
    games['result'] = [0 if row['overtime'] else row['result'] for (index, row) in games.iterrows()]

    # merge ingame_stats dataframe
    ingame_stats = pd.read_csv('data/ingame_stats.csv')
    games = games.merge(ingame_stats, 'inner', ['game_id', 'home_team', 'away_team'])

    nfldata = games[games['season'] != TEST_YEAR][[
        'away_team', 'home_team', 'away_score', 'home_score', 'total', 'result', 'overtime', 'div_game', 'game_type',
        'season', 'gameday', 'home_dsr', 'away_dsr', 'home_gc', 'home_gc_signed', 'home_gc_avg', 'home_gc_fourthqtr'
    ]]

    hah_games = nfldata.loc[(nfldata['div_game'] == 1) & (nfldata['game_type'] == 'REG')]

    hah_games['series'] = [
        "".join(sorted([row[1]['home_team'], row[1]['away_team'], str(row[1]['season'])]))
        for row in hah_games.iterrows()]

    hah_games.sort_values(['series', 'gameday'], inplace=True)

    print()
    for METRIC in metrics:
        match METRIC:
            case 'point diff':
                outcomes = [row['result'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'point ratio':
                outcomes = [row['home_score'] / row['total'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'pythagorean point ratio':
                outcomes = [row['home_score']**2 / (row['home_score']**2 + row['away_score']**2)
                            for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'dsr':
                outcomes = [row['home_dsr'] - row['away_dsr'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'game control':
                outcomes = [row['home_gc'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'game control signed':
                outcomes = [row['home_gc_signed'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'game control avg':
                outcomes = [row['home_gc_avg'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case 'game control fourth':
                outcomes = [row['home_gc_fourthqtr'] for (index, row) in hah_games.reset_index().iloc[::2, :].iterrows()]
            case _:
                outcomes = []

        lr_X = np.array(outcomes)
        lr_y = np.array([int(row['result'] < 0) for (index, row) in hah_games.reset_index().iloc[1::2, :].iterrows()])

        clf = LogisticRegression().fit(lr_X.reshape(-1, 1), lr_y.reshape(-1, 1))

        h = min(outcomes)
        for i in np.arange(min(outcomes), max(outcomes), 0.0001):
            if abs(clf.predict_proba(np.array([i]).reshape(-1, 1))[0][1] - .5) < .001:
                h = i / 2
                break

        # develop Markov Chain
        teams = sorted(list(set(hah_games['home_team'].tolist() + hah_games['away_team'].tolist())))
        P = np.zeros((len(teams), len(teams)))
        n_games = np.zeros(len(teams))
        games_test_year = games[(games['season'] == TEST_YEAR) & (games['game_type'] == 'REG')]

        for (index, row) in games_test_year.iterrows():
            home_i = teams.index(row['home_team'])
            away_i = teams.index(row['away_team'])
            match METRIC:
                case 'point diff':
                    spread = row['result']
                case 'point ratio':
                    spread = row['home_score'] / row['total']
                case 'pythagorean point ratio':
                    spread = row['home_score']**2 / (row['home_score']**2 + row['away_score']**2)
                case 'dsr':
                    spread = row['home_dsr'] - row['away_dsr']
                case 'game control':
                    spread = row['home_gc']
                case 'game control signed':
                    spread = row['home_gc_signed']
                case 'game control avg':
                    spread = row['home_gc_avg']
                case 'game control fourth':
                    spread = row['home_gc_fourthqtr']
                case _:
                    spread = 0

            r_x = clf.predict_proba(np.array(spread + h).reshape(-1, 1))[0][1]

            n_games[home_i] += 1
            n_games[away_i] += 1

            P[home_i, away_i] += 1 - r_x
            P[away_i, home_i] += r_x
            P[home_i, home_i] += r_x
            P[away_i, away_i] += 1 - r_x

        P /= n_games

        prior = np.ones(32) / 32
        steady_state = np.linalg.matrix_power(P, 1000)
        ratings = prior.dot(steady_state)
        rating_df = pd.DataFrame({
            'team': teams,
            'rating': ratings / max(ratings)
        })
        rating_df.set_index('team', inplace=True)

        # linear regression to take home field into consideration
        linr_X = np.array([
            rating_df.loc[row['home_team']]['rating'] - rating_df.loc[row['away_team']]['rating']
            for (index, row) in games_test_year.iterrows()
        ])
        linr_y = np.array([row['result'] for (index, row) in games_test_year.iterrows()])

        linr_clf = LinearRegression()
        linr_clf.fit(linr_X.reshape(-1, 1), linr_y.reshape(-1, 1))

        # get linear regression spread predictions for each regular season game of TEST_YEAR
        linr_preds[f'{METRIC} Predicted Spread'].extend([y[0] for y in linr_clf.predict([[x] for x in linr_X])])

        # loop through TEST_YEAR's playoff games and compute accuracy
        playoffs = games[(games['season'] == TEST_YEAR) & (games['game_type'] != 'REG')]
        linr_X_test = np.array([
            rating_df.loc[row['home_team']]['rating'] - rating_df.loc[row['away_team']]['rating']
            for (index, row) in playoffs.iterrows()
        ])
        linr_y_test = np.array(playoffs['result'])

        total, correct = 0, 0
        for (index, game) in playoffs.iterrows():
            home_won = True if game['result'] > 0 else False
            home_pred = linr_clf.predict(np.array([
                                        rating_df.loc[game['home_team']]['rating'] -
                                        rating_df.loc[game['away_team']]['rating']]).reshape(-1, 1))[0][0]
            game_preds[f'{METRIC} Playoff Predictions'].append(home_pred)
            correct += home_won == (home_pred > 0)
            total += 1

        print(f"{TEST_YEAR} {METRIC}")
        results[f"{METRIC} Playoff Accuracy"].append(correct / total)
        results[f"{METRIC} LinReg Playoff RMSE"].append(mse(linr_y_test, linr_clf.predict(linr_X_test.reshape(-1, 1)), squared=False))
        results[f"{METRIC} LinReg RegSzn RMSE"].append(mse(linr_y, linr_clf.predict(linr_X.reshape(-1, 1)), squared=False))

    linr_preds['Year'].extend([TEST_YEAR]*len(games_test_year))
    linr_preds['Result'].extend(games_test_year['result'])

    playoffs = games[(games['season'] == TEST_YEAR) & (games['game_type'] != 'REG')]
    total, correct = 0, 0
    for (index, game) in playoffs.iterrows():
        home_won = True if game['result'] > 0 else False
        game_preds['Spread Playoff Predictions'].append(game['spread_line'])
        game_preds['Playoff Result'].append(home_won)
        correct += home_won == (game['spread_line'] > 0)
        total += 1

    game_preds['Year'].extend([TEST_YEAR]*len(playoffs))
    game_preds['Playoff Result Spread'].extend(playoffs['result'])
    game_preds['Playoff Game Type'].extend(playoffs['game_type'])

    print(f"{TEST_YEAR} Spread")
    results["Spread Playoff Accuracy"].append(correct / total)
    results["Spread Playoff RMSE"].append(mse(np.array(playoffs['result']), np.array(playoffs['spread_line']), squared=False))
    results["Spread RegSzn RMSE"].append(mse(np.array(games_test_year['result']), np.array(games_test_year['spread_line']), squared=False))

# results_df = pd.DataFrame.from_dict(results, orient='index')
# results_df.rename(columns={y:y+2005 for y in range(len(results_df.columns))}, inplace=True)
# results_df.to_csv('results/playoff_results.csv')

pd.DataFrame.from_dict(game_preds, orient='index').to_csv('data/game_preds.csv')

# pd.DataFrame.from_dict(linr_preds, orient='index').to_csv('data/linr_preds.csv')
