"""ESP32-S3 BLE-only hypothetical energy sensitivity; NOT measured currents."""
import csv
from pathlib import Path

def scenarios():
    fixed=.3+.1+.65+.05+.5+(35+3*14)/15+8*.010+60*2*10/86400
    rows=[]
    for label,ppg in [('low_PPG',1),('example',4),('high_PPG',10)]:
        for capacity in [150,300]:
            current=fixed+ppg
            rows.append(dict(scenario='ESP32S3_BLE_'+label,battery_capacity_mAh=capacity,
                usable_fraction=.8,assumed_mean_battery_current_mA=round(current,6),
                theoretical_h=round(capacity*.8/current,4),evidence='L3 hypothetical; not measured'))
    return rows

if __name__=='__main__':
    rows=scenarios();output=Path(__file__).with_name('energy_scenarios.csv')
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row['scenario'],row['battery_capacity_mAh'],row['theoretical_h'])
