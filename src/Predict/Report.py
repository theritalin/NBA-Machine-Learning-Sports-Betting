"""Console rendering of the analysis produced by Recommender."""
from colorama import Fore, Style

from src.Predict.Recommender import (GOOD, GOOD_EV, MAX_STAKE_FRACTION, PASS, STRONG, STRONG_EV,
                                     recommended_bets)

RATING_STYLE = {STRONG: (Fore.GREEN + Style.BRIGHT, '★★★'), GOOD: (Fore.YELLOW, '★★ '), PASS: (Fore.RED, '   ')}


def _pct(value):
    return '   -  ' if value is None else f"{value * 100:5.1f}%"


def _american(odds):
    return f"{odds:+d}"


def _table(headers, rows, aligns):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def line(cells):
        return '  '.join(str(c).ljust(w) if a == 'l' else str(c).rjust(w) for c, w, a in zip(cells, widths, aligns))

    return line(headers), '  '.join('─' * w for w in widths), [line(r) for r in rows]


def print_report(analysis, model_name, sportsbook, bankroll):
    bets = recommended_bets(analysis)
    title = f" {model_name} · {sportsbook or 'elle girilen oranlar'} · kasa {bankroll:g} "
    print(Style.BRIGHT + f"\n{title:═^100}" + Style.RESET_ALL)

    print(Style.BRIGHT + Fore.CYAN + "\n▶ ÖNERİLEN BAHİSLER (en avantajlıdan başlayarak)" + Style.RESET_ALL)
    if not bets:
        print(Fore.YELLOW + "  Bugün yeterince avantajlı bahis yok. En iyi karar: OYNAMA." + Style.RESET_ALL)
    else:
        rows = [[i, b['game'], b['type'], b['selection'], f"{b['decimal_odds']:.2f} ({_american(b['odds'])})",
                 _pct(b['win_prob']), f"{b['ev_pct']:+.1f}", f"{RATING_STYLE[b['rating']][1]} {b['rating']}",
                 f"{b['stake']:g}"] for i, b in enumerate(bets, 1)]
        header, rule, lines = _table(['#', 'Maç', 'Tür', 'BAHİS', 'Oran', 'Kazanma', 'Beklenen kâr/100', 'Öneri', 'Yatır'],
                                     rows, 'rllllrrlr')
        print('  ' + Style.BRIGHT + header + Style.RESET_ALL)
        print('  ' + rule)
        for bet, text in zip(bets, lines):
            print('  ' + RATING_STYLE[bet['rating']][0] + text + Style.RESET_ALL)

    print(Style.BRIGHT + Fore.CYAN + "\n▶ TÜM MAÇLAR (model ne düşünüyor?)" + Style.RESET_ALL)
    rows = []
    for g in analysis:
        ml_bets = {b['selection']: b for b in g['bets'] if b['type'] == 'Maç Sonucu'}
        ou_bet = next((b for b in g['bets'] if b['type'] == 'Alt/Üst'), None)

        def side(team, odds, prob):
            bet = ml_bets.get(f"{team} kazanır")
            return f"{team} {_pct(prob).strip()}" + (f" [{bet['decimal_odds']:.2f}, {bet['ev_pct']:+.0f}]" if bet else '')

        ou = '-' if not g['ou_pick'] else (
            f"{g['ou_pick']} {g['total']} (güven {_pct(g['ou_confidence']).strip()}"
            + ('' if ou_bet['rating'] != PASS else ', zayıf') + ')')
        rows.append([side(g['away'], g['away_odds'], 1 - g['home_prob']), side(g['home'], g['home_odds'], g['home_prob']),
                     g['winner'], ou])
    header, rule, lines = _table(['Deplasman  olasılık [oran, kâr/100]', 'Ev sahibi  olasılık [oran, kâr/100]',
                                  'Favori', 'Alt/Üst tahmini'], rows, 'llll')
    print('  ' + Style.BRIGHT + header + Style.RESET_ALL)
    print('  ' + rule)
    for text in lines:
        print('  ' + text)

    print(Style.DIM + f"""
  Nasıl okunur?
  • Oran            : ondalık oran (parantez içinde Amerikan oranı). 2.10 = 100 yatır, kazanırsan 210 al.
  • Kazanma         : bahsin tutma ihtimali (model ile bahis şirketi görüşünün harmanı).
  • Beklenen kâr/100: bu bahsi çok kez oynasan 100 birim başına ortalama kazancın. Eksi ise uzun vadede kaybettirir.
  • Öneri           : ★★★ GÜÇLÜ = beklenen kâr ≥ %{STRONG_EV * 100:.0f}, ★★ ORTA = ≥ %{GOOD_EV * 100:.0f}. Altındakiler listelenmez (PAS).
  • Yatır           : çeyrek Kelly ile önerilen tutar (kasanın en fazla %{MAX_STAKE_FRACTION * 100:.0f}'ü).
""" + Style.RESET_ALL + Fore.YELLOW + """
  DİKKAT: Geçmiş testlerde bahis şirketinin oranları maçları modelden daha iyi tahmin etti ve bu öneriler
  uzun vadede kâr ettirmedi (bkz. README). Listeyi "modele göre en az kötü bahisler" olarak oku, garanti olarak değil.
  Model sakatlık/kadro bilgisini bilmez; oynamadan önce kontrol et.""" + Style.RESET_ALL)
