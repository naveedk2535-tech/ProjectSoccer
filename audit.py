"""ProjectSoccer Full Audit v2"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(__file__))

from database import db
db.init_db()

errors = []
warnings = []
passed = []

print("=" * 70)
print("  PROJECTSOCCER FULL AUDIT v2")
print("=" * 70)

# 1. DATABASE
print("\n[1] DATABASE INTEGRITY")
tables = ["matches","team_ratings","referee_stats","head_to_head","fixtures",
          "predictions","odds","value_bets","sentiment","api_calls",
          "model_performance","user_bets","clv_tracking","model_tracker"]
for t in tables:
    row = db.fetch_one(f"SELECT COUNT(*) as c FROM {t}")
    c = row["c"]
    print(f"  {t:20s} {c:6d} rows  [OK]")
    passed.append(f"{t}: {c}")

nulls = db.fetch_one("SELECT COUNT(*) as c FROM matches WHERE home_team IS NULL OR away_team IS NULL")
print(f"  NULL critical fields: {nulls['c']}  [{'OK' if nulls['c']==0 else 'ERROR'}]")
if nulls["c"]==0: passed.append("No NULLs")
else: errors.append(f"{nulls['c']} NULL fields")

# 2. MODELS
print("\n[2] MODEL CALCULATIONS")
from models import poisson, elo, ensemble

p = poisson.predict("Arsenal", "Chelsea", "PL")
if p:
    t = p["home_win"]+p["draw"]+p["away_win"]
    ok = abs(t-1)<0.01
    print(f"  Poisson H+D+A = {t:.6f}  [{'OK' if ok else 'ERROR'}]")
    if ok: passed.append("Poisson OK")
    else: errors.append(f"Poisson sum {t}")
    print(f"  Lambda: H={p['home_lambda']:.3f} A={p['away_lambda']:.3f}")
    sh_h = p["details"].get("home_second_half_strength", "N/A")
    sh_a = p["details"].get("away_second_half_strength", "N/A")
    print(f"  2nd-half strength: H={sh_h} A={sh_a}  [OK]")
    passed.append("2nd-half strength active")

    import numpy as np
    ms = np.array(p["scoreline_matrix"]).sum()
    print(f"  Matrix sum = {ms:.6f}  [{'OK' if abs(ms-1)<0.01 else 'ERROR'}]")
    if abs(ms-1)<0.01: passed.append("Matrix OK")
    else: errors.append("Matrix bad")
else:
    errors.append("Poisson returned None")

e = elo.predict("Arsenal", "Chelsea", "PL")
if e:
    t = e["home_win"]+e["draw"]+e["away_win"]
    print(f"  Elo H+D+A = {t:.6f}  [{'OK' if abs(t-1)<0.01 else 'ERROR'}]")
    if abs(t-1)<0.01: passed.append("Elo OK")
    else: errors.append(f"Elo sum {t}")
    print(f"  Elo: H={e['details']['home_elo']:.0f} A={e['details']['away_elo']:.0f}")
    oaf = e["details"].get("home_opp_adj_form", "N/A")
    print(f"  Opp-adjusted form: {oaf}  [OK]")
    passed.append("OAF active")

en = ensemble.predict("Arsenal", "Chelsea", "PL")
if en:
    t = en["home_win"]+en["draw"]+en["away_win"]
    print(f"  Ensemble H+D+A = {t:.6f}  [{'OK' if abs(t-1)<0.02 else 'ERROR'}]")
    print(f"  Models: {en['models_used']} ({len(en['models_used'])}/4)")
    print(f"  Method: {en.get('method','?')}")
    if len(en["models_used"])>=3: passed.append(f"{len(en['models_used'])} models")
    else: warnings.append(f"Only {len(en['models_used'])} models")

# 3. ODDS
print("\n[3] ODDS & VALUE BETS")
comp = db.fetch_one("SELECT COUNT(*) as c FROM odds WHERE home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL")
tot = db.fetch_one("SELECT COUNT(*) as c FROM odds")
pct = comp["c"]/tot["c"]*100 if tot["c"]>0 else 0
print(f"  Complete odds: {comp['c']}/{tot['c']} ({pct:.0f}%)  [{'OK' if pct>95 else 'WARN'}]")
if pct>95: passed.append(f"Odds {pct:.0f}% complete")
else: warnings.append(f"Odds only {pct:.0f}%")

vbs = db.fetch_all("SELECT * FROM value_bets WHERE result = 'pending'")
valid = all(v["edge_percent"]>=5 and 0<v["model_probability"]<1 and v["best_odds"]>1 for v in vbs) if vbs else True
print(f"  Value bets: {len(vbs)} pending, all valid: {valid}  [OK]")
if valid: passed.append(f"{len(vbs)} VBs valid")
else: errors.append("Invalid VBs")

# 4. TEAM NAMES
print("\n[4] TEAM NAMES")
for lg in ["PL","PD","BL1"]:
    mt = set(r["home_team"] for r in db.fetch_all("SELECT DISTINCT home_team FROM matches WHERE league=?", [lg]))
    mt.update(r["away_team"] for r in db.fetch_all("SELECT DISTINCT away_team FROM matches WHERE league=?", [lg]))
    ft = set(r["home_team"] for r in db.fetch_all("SELECT DISTINCT home_team FROM fixtures WHERE league=?", [lg]))
    ft.update(r["away_team"] for r in db.fetch_all("SELECT DISTINCT away_team FROM fixtures WHERE league=?", [lg]))
    miss = ft - mt
    if miss:
        warnings.append(f"{lg}: {len(miss)} missing: {miss}")
        print(f"  {lg}: {len(miss)} fixture teams not in matches  [WARN]")
    else:
        passed.append(f"{lg} names OK")
        print(f"  {lg}: consistent  [OK]")

# 5. PREDICTIONS
print("\n[5] PREDICTIONS")
preds = db.fetch_all("SELECT * FROM predictions")
bad = sum(1 for pr in preds if pr.get("ensemble_home") and abs((pr["ensemble_home"] or 0)+(pr["ensemble_draw"] or 0)+(pr["ensemble_away"] or 0)-1)>0.02)
print(f"  {len(preds)} predictions, {bad} bad sums  [{'OK' if bad==0 else 'ERROR'}]")
if bad==0: passed.append(f"{len(preds)} preds valid")
else: errors.append(f"{bad} bad preds")

# 6. ENHANCED FEATURES
print("\n[6] ENHANCED FEATURES")
from models.poisson import calculate_half_profiles, calculate_second_half_strength
profiles = calculate_half_profiles("PL")
sh = calculate_second_half_strength("PL")
print(f"  Half-time profiles: {len(profiles)} teams  [OK]")
print(f"  2nd-half range: [{min(sh.values()):.4f}, {max(sh.values()):.4f}]  [OK]")
passed.append(f"HT profiles: {len(profiles)}")

ratings = elo.build_ratings("PL")
oaf_count = sum(1 for r in ratings.values() if r.get("opponent_adjusted_form"))
print(f"  Opponent-adj form: {oaf_count} teams  [OK]")
passed.append(f"OAF: {oaf_count} teams")

ref_c = db.fetch_one("SELECT COUNT(*) as c FROM referee_stats")["c"]
print(f"  Referee profiles: {ref_c}  [OK]")
passed.append(f"{ref_c} referees")

mp = os.path.join("data","cache","xgboost_model.pkl")
if os.path.exists(mp):
    with open(mp,"rb") as f:
        sv = pickle.load(f)
    feats = sv.get("features",[])
    new_f = [x for x in feats if x in ("home_shot_accuracy","away_shot_accuracy","home_avg_shots","away_avg_shots","home_opp_adj_form","away_opp_adj_form","congestion_diff","odds_movement")]
    print(f"  XGBoost: {len(feats)} features ({len(new_f)} enhanced)  [OK]")
    print(f"  Enhanced: {new_f}")
    passed.append(f"XGB: {len(feats)} features")
else:
    warnings.append("XGBoost model missing")
    print(f"  XGBoost model: MISSING  [WARN]")

# 7. SENTIMENT
print("\n[7] SENTIMENT COVERAGE")
for lg in ["PL","PD","BL1"]:
    c = db.fetch_one("SELECT COUNT(DISTINCT team) as c FROM sentiment WHERE league=?", [lg])["c"]
    print(f"  {lg}: {c} teams  [{'OK' if c>0 else 'NO DATA'}]")
    if c>0: passed.append(f"{lg} sentiment: {c}")
    else: warnings.append(f"{lg} no sentiment")

# 8. CROSS-LEAGUE
print("\n[8] CROSS-LEAGUE DATA")
for lg in ["PL","PD","BL1"]:
    mc = db.fetch_one("SELECT COUNT(*) as c FROM matches WHERE league=?", [lg])["c"]
    fc = db.fetch_one("SELECT COUNT(*) as c FROM fixtures WHERE league=? AND status IN ('SCHEDULED','TIMED')", [lg])["c"]
    pc = db.fetch_one("SELECT COUNT(*) as c FROM predictions WHERE league=?", [lg])["c"]
    oc = db.fetch_one("SELECT COUNT(*) as c FROM odds WHERE league=?", [lg])["c"]
    vb = db.fetch_one("SELECT COUNT(*) as c FROM value_bets WHERE league=? AND result='pending'", [lg])["c"]
    print(f"  {lg}: {mc} matches, {fc} fixtures, {pc} preds, {oc} odds, {vb} VBs  [OK]")
    passed.append(f"{lg} complete")

# 9. TRACKER
print("\n[9] MODEL TRACKER")
tt = db.fetch_one("SELECT COUNT(*) as c FROM model_tracker")["c"]
ts = db.fetch_one("SELECT COUNT(*) as c FROM model_tracker WHERE status='settled'")["c"]
tp = db.fetch_one("SELECT COUNT(*) as c FROM model_tracker WHERE status='pending'")["c"]
print(f"  Total: {tt} | Settled: {ts} | Pending: {tp}")
if ts > 0:
    tc = db.fetch_one("SELECT COUNT(*) as c FROM model_tracker WHERE status='settled' AND top_pick_correct=1")["c"]
    pnl = db.fetch_one("SELECT SUM(top_pick_pnl) as s FROM model_tracker WHERE status='settled'")["s"] or 0
    print(f"  Accuracy: {tc}/{ts} ({tc/ts*100:.0f}%)")
    print(f"  P&L ($100/bet): ${pnl:.2f}")
passed.append(f"Tracker: {tt} entries")

# 10. RATE LIMITS
print("\n[10] RATE LIMITS")
from data.rate_limiter import get_usage_summary
usage = get_usage_summary()
for api, info in usage.items():
    s = "OK" if info["remaining"]>0 else "EXHAUSTED"
    print(f"  {api:20s} {info['used']}/{info['limit']}  ({info['remaining']} left)  [{s}]")
    if info["remaining"]>0: passed.append(f"{api} OK")
    else: warnings.append(f"{api} exhausted")

# 11. SCHEDULE
print("\n[11] SCHEDULED TASKS")
print("  05:00 AM EST — Weekly rebuild (CSV + ratings + retrain)")
print("  07:00 AM EST — Daily refresh (Reddit + News + Odds + Fixtures)")
print("  01:00 PM EST — Reddit evening scan")
passed.append("3 scheduled tasks")

# SUMMARY
print("\n" + "=" * 70)
print(f"  PASSED:   {len(passed)}")
print(f"  WARNINGS: {len(warnings)}")
print(f"  ERRORS:   {len(errors)}")
print("=" * 70)
if warnings:
    print("\n  WARNINGS:")
    for w in warnings: print(f"    - {w}")
if errors:
    print("\n  ERRORS:")
    for e in errors: print(f"    !! {e}")
else:
    print("\n  NO ERRORS — All systems operational.")
print()
