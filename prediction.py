import sys
import pandas as pd
import joblib

model = joblib.load('rf_final_model.pkl')
threshold = joblib.load('rf_threshold.pkl')

FEATURES = ['恶性值', '坏死值', '肾盂积水值', '浸润值', '形状值', '红细胞计数(RBC#)-尿液']

PROMPTS = [
    ('恶性值',               'range 0-2'),
    ('坏死值',               'range 0-1'),
    ('肾盂积水值',           'range 0-1'),
    ('浸润值',               'range 0-1'),
    ('形状值',               'range 0-1'),
    ('红细胞计数(RBC#)-尿液', 'normal 3.5-7.0'),
]


def risk_label(prob):
    if prob < 0.3:  return 'Low',       'Routine follow-up'
    if prob < 0.5:  return 'Moderate',  'Close follow-up'
    if prob < 0.7:  return 'High',      'Enhanced monitoring'
    return              'Very High',    'Immediate intervention'


def predict(values: dict) -> dict:
    prob = model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1]
    risk, advice = risk_label(prob)
    return {
        'probability': f'{prob:.1%}',
        'prediction':  'Recurrence' if prob >= threshold else 'No recurrence',
        'risk':        risk,
        'advice':      advice,
    }


def batch(input_csv, output_csv='predictions.csv'):
    df = pd.read_csv(input_csv, encoding='utf-8')
    probs = model.predict_proba(df[FEATURES])[:, 1]
    df['probability'] = [f'{p:.1%}' for p in probs]
    df['prediction']  = ['Recurrence' if p >= threshold else 'No recurrence' for p in probs]
    df['risk']        = [risk_label(p)[0] for p in probs]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'Saved to {output_csv}')


def interactive():
    print(f'\nRecurrence Risk Predictor  |  AUC 0.7136  |  threshold {threshold:.4f}')
    while True:
        values = {}
        for name, hint in PROMPTS:
            raw = input(f'  {name} ({hint}): ').strip()
            if raw.lower() == 'q':
                return
            values[name] = float(raw)
        result = predict(values)
        print('\n  ' + '  '.join(f'{k}: {v}' for k, v in result.items()) + '\n')
        if input('Next patient? (y/n): ').strip().lower() != 'y':
            return


if __name__ == '__main__':
    if len(sys.argv) == 2:
        batch(sys.argv[1])
    else:
        interactive()