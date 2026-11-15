import json
import os
import subprocess
import sys
import time
from datetime import date
from functools import lru_cache

from flask import Flask, render_template, request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SPORTSBOOKS = ['fanduel', 'draftkings', 'betmgm']


@lru_cache(maxsize=32)
def fetch_game_data(sportsbook, bankroll, ttl_hash=None):
    """Runs the predictor for one sportsbook. Returns (data, error_message)."""
    del ttl_hash
    cmd = [sys.executable, "main.py", "-xgb", f"-odds={sportsbook}", "-kasa", str(bankroll), "-json"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return None, (result.stderr.strip().splitlines() or ["Bilinmeyen hata"])[-1]
    return json.loads(result.stdout).get('XGBoost'), result.stderr.strip()


def get_ttl_hash(seconds=600):
    """Return the same value within `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)


@app.route("/")
def index():
    bankroll = request.args.get('kasa', default=1000, type=float)
    books = {}
    for sportsbook in SPORTSBOOKS:
        data, message = fetch_game_data(sportsbook, bankroll, ttl_hash=get_ttl_hash())
        books[sportsbook] = {'data': data, 'message': message}
    return render_template('index.html', today=date.today(), books=books, bankroll=bankroll)
