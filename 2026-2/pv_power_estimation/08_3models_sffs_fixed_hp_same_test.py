"""
[Full69 하이퍼파라미터 고정 실험]

- RF, LightGBM, LSTM의 Full69 모델에서
  선정한 하이퍼파라미터를 사용한다.
- 특징 조합이 변경되어도 하이퍼파라미터를
  다시 탐색하지 않는다.
- 선정한 특징으로 모델 가중치는 새로 학습한다.
- 동일한 공통 Test 시점에서 성능을 평가한다.
- Optuna를 이용한 재최적화는 수행하지 않는다.

※ 고정하는 것은 하이퍼파라미터이며,
  모델 가중치가 고정되는 것은 아니다.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics(df):
    y, p, c = (df[k].to_numpy(dtype=float) for k in ('true_MWh','pred_MWh','capacity_MW'))
    e = p-y
    return dict(n_samples=len(df), MAE_MWh=float(np.mean(abs(e))),
                RMSE_MWh=float(np.sqrt(np.mean(e*e))),
                NMAE_pct=float(np.mean(abs(e)/c)*100),
                NRMSE_pct=float(np.sqrt(np.mean((e/c)**2))*100),
                MBE_MWh=float(e.mean()), R2=float(r2_score(y,p)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--models', default='random_forest,lightgbm,lstm')
    a.add_argument('--data', default='chungbuk_pv_features.csv')
    a.add_argument('--base_script', default='04_exp1_train_v1.py')
    a.add_argument('--rf_script', default='04_exp1_train_random_forest.py')
    a.add_argument('--sffs_dir', default='pv_exp5_shap_sffs/results')
    a.add_argument('--reference', default='pv_training_outputs_final_comparison/common_test_timestamps_C6_Full_3models.csv')
    a.add_argument('--output_dir', default='pv_exp7_3models_sffs_fixed_hp')
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--epochs', type=int, default=200)
    a.add_argument('--patience', type=int, default=25)
    a.add_argument('--rf_n_jobs', type=int, default=-1)
    a.add_argument('--device', default='auto')
    a.add_argument('--no_clip', action='store_true')
    args = a.parse_args()
    names = [x.strip() for x in args.models.split(',') if x.strip()]
    allowed = ('random_forest','lightgbm','lstm')
    if not names or len(names)!=len(set(names)) or any(x not in allowed for x in names):
        a.error('Invalid --models. Use random_forest,lightgbm,lstm')
    full_paths = {'random_forest':'pv_training_outputs_rf/models/random_forest/C6_Full/model.pkl',
                  'lightgbm':'pv_training_outputs/models/lightgbm/C6_Full/model.pkl',
                  'lstm':'pv_training_outputs/models/lstm/C6_Full/model.pt'}
    inputs = [args.data,args.base_script,args.reference] + ([args.rf_script] if 'random_forest' in names else [])
    inputs += [full_paths[m] for m in names]
    inputs += [str(Path(args.sffs_dir)/f'features_sffs_{m}.csv') for m in names]
    missing = [p for p in inputs if not Path(p).is_file()]
    if missing:
        a.error('Missing input files:\n'+'\n'.join(missing))
    base = load_module(Path(args.base_script),'fixed_base04')
    rfmod = load_module(Path(args.rf_script),'fixed_rf04') if 'random_forest' in names else None
    base.set_seed(args.seed)
    data = base.read_feature_csv(args.data)
    train, valid, test = base.split_by_year(data)
    ref = pd.read_csv(args.reference,encoding='utf-8-sig')
    dt = base.DATE_COL
    times = pd.DatetimeIndex(pd.to_datetime(ref[dt],errors='raise'))
    if len(times)==0 or times.has_duplicates or times.isna().any():
        raise ValueError('Reference timestamps are empty, duplicated or invalid')
    times = times.sort_values()
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for name in names:
        print(f'\n=== {name} ===',flush=True)
        feat_file=Path(args.sffs_dir)/f'features_sffs_{name}.csv'
        feat_df=pd.read_csv(feat_file,encoding='utf-8-sig')
        if 'feature' not in feat_df.columns:
            raise ValueError(f'{feat_file}: feature column missing')
        features=feat_df['feature'].dropna().astype(str).tolist()
        if not features or len(features)!=len(set(features)):
            raise ValueError(f'{name}: empty or duplicated features')
        absent=set(features)-set(data.columns)
        if absent: raise ValueError(f'{name}: missing data features {sorted(absent)}')
        model_out=out/name
        model_out.mkdir(exist_ok=True)
        case='SFFS_FixedFull69HP'
        if name=='lstm':
            import torch
            bundle=torch.load(full_paths[name],map_location='cpu',weights_only=False)
            hp=dict(bundle['best_params'])
            required=['seq_len','hidden_size','num_layers','dropout','learning_rate','batch_size']
            if any(k not in hp for k in required):
                raise ValueError(f'LSTM best_params missing: {set(required)-set(hp)}')
            base.set_seed(args.seed)
            device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
            scaler=StandardScaler().fit(train[features].dropna().astype(float).values)
            seq_len=int(hp['seq_len'])
            Xtr,ytr,_=base.make_sequences_gap_safe(train,features,seq_len,scaler=scaler)
            Xva,yva,tva=base.make_sequences_gap_safe(valid,features,seq_len,scaler=scaler)
            Xte,yte,tte=base.make_sequences_gap_safe(test,features,seq_len,scaler=scaler)
            if not all(map(len,(Xtr,Xva,Xte))): raise ValueError('Empty LSTM sequences')
            mdl=base.LSTMRegressor(len(features),int(hp['hidden_size']),int(hp['num_layers']),float(hp['dropout']))
            mdl,best_score,_=base.train_torch_one_run(
                mdl,Xtr,ytr,Xva,yva,valid,tva,lr=float(hp['learning_rate']),
                batch_size=int(hp['batch_size']),max_epochs=args.epochs,
                patience=args.patience,device=device,clip_predictions=not args.no_clip)
            pv=base.predict_torch(mdl,Xva,int(hp['batch_size']),device)
            pt=base.predict_torch(mdl,Xte,int(hp['batch_size']),device)
            val_df=base.build_prediction_df(valid,tva,pv,'LSTM',case,not args.no_clip)
            raw=base.build_prediction_df(test,tte,pt,'LSTM',case,not args.no_clip)
            torch.save({'state_dict':mdl.cpu().state_dict(),'best_params':hp,'feature_cols':features},model_out/'model.pt')
            joblib.dump(scaler,model_out/'scaler.pkl')
        else:
            bundle=joblib.load(full_paths[name]); hp=dict(bundle['best_params'])
            mod=rfmod if name=='random_forest' else base
            use_cols=list(dict.fromkeys(features+[base.TARGET_COL,base.DATE_COL,base.CAPACITY_COL,base.POWER_COL,base.DAYTIME_COL]+([base.SEGMENT_COL] if base.SEGMENT_COL in data.columns else [])))
            tr=train[use_cols].dropna().copy(); va=valid[use_cols].dropna().copy(); te=test[use_cols].dropna().copy()
            if any(x.empty for x in (tr,va,te)): raise ValueError(f'{name}: empty split')
            if name=='random_forest':
                params={**hp,'random_state':args.seed,'n_jobs':args.rf_n_jobs}
                params.setdefault('bootstrap',True); params.setdefault('criterion','squared_error')
                mdl=RandomForestRegressor(**params)
                mdl.fit(tr[features],tr[base.TARGET_COL])
                pv=mdl.predict(va[features]); pt=mdl.predict(te[features])
                val_df=rfmod.build_prediction_df(va,pv,'RandomForest',case,not args.no_clip)
                raw=rfmod.build_prediction_df(te,pt,'RandomForest',case,not args.no_clip)
            else:
                import lightgbm as lgb
                params={'objective':'regression','metric':'rmse','random_state':args.seed,'verbosity':-1,**hp}
                mdl=lgb.LGBMRegressor(**params)
                mdl.fit(tr[features],tr[base.TARGET_COL],eval_set=[(va[features],va[base.TARGET_COL])],eval_metric='rmse',callbacks=[lgb.early_stopping(100,verbose=False)])
                pv=mdl.predict(va[features],num_iteration=mdl.best_iteration_)
                pt=mdl.predict(te[features],num_iteration=mdl.best_iteration_)
                val_df=base.build_prediction_df(va,va[dt].values,pv,'LightGBM',case,not args.no_clip)
                raw=base.build_prediction_df(te,te[dt].values,pt,'LightGBM',case,not args.no_clip)
            joblib.dump({'model':mdl,'best_params':hp,'feature_cols':features},model_out/'model.pkl')
        raw[dt]=pd.to_datetime(raw[dt],errors='raise')
        if raw[dt].duplicated().any(): raise ValueError(f'{name}: duplicate predictions')
        indexed=raw.set_index(dt)
        absent=times.difference(pd.DatetimeIndex(indexed.index))
        if len(absent): raise ValueError(f'{name}: {len(absent)} common timestamps missing: {list(absent[:5])}')
        aligned=indexed.loc[times].reset_index()
        if len(aligned)!=len(times): raise ValueError('Test alignment mismatch')
        for col in ['true_MWh','pred_MWh',base.CAPACITY_COL,base.DAYTIME_COL]:
            if not np.isfinite(pd.to_numeric(aligned[col]).to_numpy(dtype=float)).all():
                raise ValueError(f'{name}: invalid values in {col}')
        if (aligned[base.CAPACITY_COL]<=0).any(): raise ValueError('Nonpositive capacity')
        # Rename only for local metrics, preserve original prediction CSV schema.
        def calc(frame):
            return metrics(frame.rename(columns={base.CAPACITY_COL:'capacity_MW'}))
        scopes={'ALL':aligned,'DAY':aligned.loc[aligned[base.DAYTIME_COL]>0]}
        if scopes['DAY'].empty: raise ValueError('No daytime common samples')
        valid_metrics=base.evaluate_prediction_df(val_df)
        for scope,frame in scopes.items():
            summary.append({'model':name,'method':case,'feature_count':len(features),'scope':scope,**calc(frame)})
        val_df.to_csv(model_out/'valid_predictions.csv',index=False,encoding='utf-8-sig')
        raw.to_csv(model_out/'test_predictions_raw.csv',index=False,encoding='utf-8-sig')
        aligned.to_csv(model_out/'COMMON_TEST_predictions.csv',index=False,encoding='utf-8-sig')
        (model_out/'run_info.json').write_text(json.dumps({'features':features,'fixed_full69_params':hp,'seed':args.seed,'validation_metrics':valid_metrics,'reference':args.reference,'common_test_count':len(times)},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        pd.DataFrame(summary).to_csv(out/'FINAL_SAME_TEST_metrics.csv',index=False,encoding='utf-8-sig')
        print(f'{name}: {len(features)} features, {len(times)} common timestamps, DAY NMAE={summary[-1]["NMAE_pct"]:.4f}%')
    print('\nSaved:',out.resolve())

if __name__=='__main__': main()
