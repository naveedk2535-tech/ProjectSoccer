"""
ProjectSoccer Backtester
Run predictions against historical data to validate model performance.

Usage:
    python backtest.py --league PL --seasons 2324,2425
"""
import argparse
import logging
import json
from datetime import datetime

import numpy as np

from database import db
from models import poisson, elo, ensemble
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")


def brier_score(predicted_probs, actual_outcome):
    """Calculate Brier score for a single prediction. Lower is better."""
    # actual_outcome: [1,0,0] for home win, [0,1,0] for draw, [0,0,1] for away
    return sum((p - a) ** 2 for p, a in zip(predicted_probs, actual_outcome))


def run_backtest(league="PL", seasons=None, start_from=0.4):
    """
    Run backtest on historical matches.
    start_from: fraction of matches to skip (warmup period for model).
    """
    db.init_db()

    query = "SELECT * FROM matches WHERE league = ? AND ft_result IS NOT NULL ORDER BY match_date ASC"
    params = [league]
    matches = db.fetch_all(query, params)

    if seasons:
        matches = [m for m in matches if m["season"] in seasons]

    if not matches:
        logger.error("No matches found for backtest")
        return

    skip = int(len(matches) * start_from)
    test_matches = matches[skip:]
    logger.info("Backtesting on %d matches (skipping %d warmup)", len(test_matches), skip)

    # Track per-model performance
    results = {
        "poisson": {"correct": 0, "brier": [], "total": 0},
        "elo": {"correct": 0, "brier": [], "total": 0},
        "ensemble": {"correct": 0, "brier": [], "total": 0},
    }

    # Track calibration bins
    calibration = {i/10: {"predicted": 0, "actual": 0, "count": 0} for i in range(1, 10)}

    # Value bet tracking
    value_bet_results = {"total": 0, "wins": 0, "profit": 0, "staked": 0}

    # CLV (Closing Line Value) tracking
    clv_records = []
    clv_stats = {"total": 0, "positive_clv": 0, "total_clv_pct": 0.0}

    for i, m in enumerate(test_matches):
        home = m["home_team"]
        away = m["away_team"]
        actual = m["ft_result"]
        actual_vec = [1, 0, 0] if actual == "H" else ([0, 1, 0] if actual == "D" else [0, 0, 1])

        # Poisson prediction
        p_pred = poisson.predict(home, away, league)
        if p_pred:
            probs = [p_pred["home_win"], p_pred["draw"], p_pred["away_win"]]
            predicted = ["H", "D", "A"][np.argmax(probs)]
            results["poisson"]["total"] += 1
            if predicted == actual:
                results["poisson"]["correct"] += 1
            results["poisson"]["brier"].append(brier_score(probs, actual_vec))

        # Elo prediction
        e_pred = elo.predict(home, away, league)
        if e_pred:
            probs = [e_pred["home_win"], e_pred["draw"], e_pred["away_win"]]
            predicted = ["H", "D", "A"][np.argmax(probs)]
            results["elo"]["total"] += 1
            if predicted == actual:
                results["elo"]["correct"] += 1
            results["elo"]["brier"].append(brier_score(probs, actual_vec))

        # Ensemble (Poisson + Elo, XGBoost may not be trained yet)
        # For backtest we use a simplified ensemble
        if p_pred and e_pred:
            ens_probs = [
                p_pred["home_win"] * 0.55 + e_pred["home_win"] * 0.45,
                p_pred["draw"] * 0.55 + e_pred["draw"] * 0.45,
                p_pred["away_win"] * 0.55 + e_pred["away_win"] * 0.45,
            ]
            predicted = ["H", "D", "A"][np.argmax(ens_probs)]
            results["ensemble"]["total"] += 1
            if predicted == actual:
                results["ensemble"]["correct"] += 1
            results["ensemble"]["brier"].append(brier_score(ens_probs, actual_vec))

            # Calibration tracking
            max_prob = max(ens_probs)
            bin_key = round(max_prob, 1)
            if bin_key in calibration:
                calibration[bin_key]["count"] += 1
                calibration[bin_key]["predicted"] += max_prob
                if predicted == actual:
                    calibration[bin_key]["actual"] += 1

            # Value bet simulation using historical Pinnacle odds
            if m.get("pinnacle_home") and m.get("pinnacle_draw") and m.get("pinnacle_away"):
                pin_odds = [m["pinnacle_home"], m["pinnacle_draw"], m["pinnacle_away"]]
                pin_implied = [1/o for o in pin_odds]
                pin_total = sum(pin_implied)
                pin_true = [p/pin_total for p in pin_implied]

                bet_types = ["home_win", "draw", "away_win"]
                for idx, (model_p, bookie_p, odds) in enumerate(zip(ens_probs, pin_true, pin_odds)):
                    edge = (model_p - bookie_p) * 100
                    if edge >= config.VALUE_BET_SETTINGS["min_edge_percent"]:
                        # Simulate bet
                        stake = 1.0  # unit stake
                        value_bet_results["total"] += 1
                        value_bet_results["staked"] += stake
                        if actual_vec[idx] == 1:
                            value_bet_results["wins"] += 1
                            value_bet_results["profit"] += (odds - 1) * stake
                        else:
                            value_bet_results["profit"] -= stake

                    # CLV tracking: compare model prob vs Pinnacle closing implied
                    # Pinnacle closing odds are treated as the closing line here
                    pinnacle_closing_implied = pin_true[idx]
                    if model_p > 0 and pinnacle_closing_implied > 0:
                        clv_pct = ((model_p / pinnacle_closing_implied) - 1) * 100
                        clv_stats["total"] += 1
                        clv_stats["total_clv_pct"] += clv_pct
                        if clv_pct > 0:
                            clv_stats["positive_clv"] += 1
                        clv_records.append({
                            "league": league,
                            "match_date": m["match_date"],
                            "home_team": home,
                            "away_team": away,
                            "bet_type": bet_types[idx],
                            "model_probability": round(model_p, 4),
                            "pinnacle_closing_implied": round(pinnacle_closing_implied, 4),
                            "clv_percent": round(clv_pct, 2),
                        })

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d matches", i + 1, len(test_matches))

    # Print results
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — {league}")
    print(f"  {len(test_matches)} matches tested")
    print("=" * 60)

    for model_name, data in results.items():
        if data["total"] == 0:
            continue
        acc = data["correct"] / data["total"]
        avg_brier = sum(data["brier"]) / len(data["brier"]) if data["brier"] else 0
        print(f"\n  {model_name.upper()}")
        print(f"    Accuracy:    {acc*100:.1f}%  ({data['correct']}/{data['total']})")
        print(f"    Brier Score: {avg_brier:.4f}  (lower is better, 0.25 = random)")

        # Save to database
        db.upsert("model_performance", {
            "league": league,
            "season": "backtest",
            "model_name": model_name,
            "total_predictions": data["total"],
            "correct_predictions": data["correct"],
            "accuracy": round(acc, 4),
            "brier_score": round(avg_brier, 4),
        }, ["league", "season", "model_name"])

    # Value bet results
    if value_bet_results["total"] > 0:
        roi = (value_bet_results["profit"] / value_bet_results["staked"]) * 100
        yield_pct = value_bet_results["profit"] / value_bet_results["total"] * 100
        win_rate = value_bet_results["wins"] / value_bet_results["total"] * 100
        print(f"\n  VALUE BETS (edge >= {config.VALUE_BET_SETTINGS['min_edge_percent']}%)")
        print(f"    Total bets:  {value_bet_results['total']}")
        print(f"    Win rate:    {win_rate:.1f}%")
        print(f"    ROI:         {roi:+.1f}%")
        print(f"    Yield:       {yield_pct:+.1f}%")
        print(f"    Profit:      {value_bet_results['profit']:+.2f} units")
    else:
        print("\n  No value bets found in backtest period")

    # CLV results
    if clv_stats["total"] > 0:
        avg_clv = clv_stats["total_clv_pct"] / clv_stats["total"]
        positive_pct = (clv_stats["positive_clv"] / clv_stats["total"]) * 100
        print(f"\n  CLOSING LINE VALUE (CLV)")
        print(f"    Total comparisons: {clv_stats['total']}")
        print(f"    Avg CLV:           {avg_clv:+.2f}%")
        print(f"    Positive CLV rate: {positive_pct:.1f}%")
        if avg_clv > 0:
            print(f"    Verdict:           Model has GENUINE EDGE (beating closing line)")
        else:
            print(f"    Verdict:           Model does NOT beat closing line consistently")

        # Save CLV records to database
        try:
            if clv_records:
                db.insert_many("clv_tracking", clv_records)
                logger.info("Saved %d CLV records to database", len(clv_records))
        except Exception as e:
            logger.warning("Could not save CLV records: %s", e)
    else:
        print("\n  No CLV data available (no Pinnacle odds in dataset)")

    # Calibration
    print(f"\n  CALIBRATION")
    for bin_val in sorted(calibration.keys()):
        data = calibration[bin_val]
        if data["count"] > 0:
            observed = data["actual"] / data["count"] * 100
            print(f"    {bin_val*100:.0f}% predicted → {observed:.1f}% observed  (n={data['count']})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProjectSoccer Backtester")
    parser.add_argument("--league", default="PL", help="League code")
    parser.add_argument("--seasons", default=None, help="Comma-separated seasons (e.g., 2324,2425)")
    args = parser.parse_args()

    seasons = args.seasons.split(",") if args.seasons else None
    run_backtest(args.league, seasons)
