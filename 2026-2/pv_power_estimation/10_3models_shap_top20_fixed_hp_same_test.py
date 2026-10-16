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
    a.add_argument('--base_script', default='04_exp1_train.py')
    a.add_argument('--rf_script', default='04_exp1_train_random_forest.py')
    a.add_argument('--sffs_dir', default='pv_exp5_shap_sffs')
    a.add_argument('--shap_root', default='pv_shap_lgbm_rf_lstm_outputs')
    a.add_argument('--exp17_dir', default='pv_exp7_3models_sffs_fixed_hp')
    a.add_argument('--expected_test_count', type=int, default=7872)
    a.add_argument('--reference', default='pv_training_outputs_final_comparison/common_test_timestamps_C6_Full_3models.csv')
    a.add_argument('--output_dir', default='pv_exp9_3models_shap_top20_fixed_hp')
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
    inputs += [str(Path(args.sffs_dir)/'sffs'/m/'candidate_SHAP_Top20.csv') for m in names]
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
    if args.expected_test_count > 0 and len(times) != args.expected_test_count:
        raise ValueError(f'Expected {args.expected_test_count} common Test timestamps, found {len(times)}')
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for name in names:
        print(f'\n=== {name} ===',flush=True)
        feat_file=Path(args.sffs_dir)/'sffs'/name/'candidate_SHAP_Top20.csv'
        feat_df=pd.read_csv(feat_file,encoding='utf-8-sig')
        if 'feature' not in feat_df.columns:
            raise ValueError(f'{feat_file}: feature column missing')
        features=feat_df['feature'].dropna().astype(str).tolist()
        if len(features)!=20 or len(set(features))!=20:
            raise ValueError(f'{name}: expected exactly 20 distinct SHAP candidates; found {len(features)}')
        # Audit: the same ranking as the SHAP experiment, if that source still exists.
        shap_path=Path(args.shap_root)/name/'shap_feature_importance.csv'
        if shap_path.is_file():
            rank=pd.read_csv(shap_path,encoding='utf-8-sig')
            if 'feature' not in rank: raise ValueError(f'{shap_path}: feature column missing')
            if 'rank' in rank: rank=rank.sort_values('rank')
            elif 'mean_abs_shap' in rank: rank=rank.sort_values('mean_abs_shap',ascending=False)
            elif 'normalized_importance_pct' in rank: rank=rank.sort_values('normalized_importance_pct',ascending=False)
            else: raise ValueError(f'{shap_path}: cannot identify ranking column')
            full=list(base.build_feature_cases(base.build_feature_groups(data))['C6_Full'])
            expected=(rank.drop_duplicates('feature')
                      .loc[lambda x:x['feature'].isin(full),'feature'].astype(str).head(20).tolist())
            if expected!=features:
                raise ValueError(f'{name}: SFFS candidate Top20 differs from Full69 SHAP Top20')
        full=set(base.build_feature_cases(base.build_feature_groups(data))['C6_Full'])
        extra=set(features)-full
        if extra: raise ValueError(f'{name}: features outside Full69: {sorted(extra)}')
        absent=set(features)-set(data.columns)
        if absent: raise ValueError(f'{name}: missing data features {sorted(absent)}')
        model_out=out/name
        model_out.mkdir(exist_ok=True)
        case='SHAP_Top20_FixedFull69HP'
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
        (model_out/'run_info.json').write_text(json.dumps({
            'features':features,'feature_source_file':str(feat_file),
            'feature_source':'original SFFS SHAP Top20 candidate pool',
            'fixed_full69_params':hp,'full69_model_path':full_paths[name],
            'seed':args.seed,'epochs':args.epochs,'patience':args.patience,
            'validation_metrics':valid_metrics,'reference':args.reference,
            'common_test_count':len(times),'test_timestamp_check':'strict',
            'no_optuna':True},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        pd.DataFrame({'feature_rank':range(1,21),'feature':features}).to_csv(
            model_out/'SHAP_Top20_features.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(summary).to_csv(out/'FINAL_SAME_TEST_metrics.csv',index=False,encoding='utf-8-sig')
        print(f'{name}: {len(features)} features, {len(times)} common timestamps, DAY NMAE={summary[-1]["NMAE_pct"]:.4f}%')
    # Optional descriptive comparison with experiment 17 (test NOT used for tuning).
    previous=Path(args.exp17_dir)/'FINAL_SAME_TEST_metrics.csv'
    if previous.is_file():
        # Verify the earlier SFFS diagonal predictions refer to identical Test rows.
        for name in names:
            pred_path=Path(args.exp17_dir)/name/'COMMON_TEST_predictions.csv'
            if not pred_path.is_file():
                raise FileNotFoundError(f'Exp17 comparison requires: {pred_path}')
            prior=pd.read_csv(pred_path,encoding='utf-8-sig')
            prior[dt]=pd.to_datetime(prior[dt],errors='raise')
            prior_times=pd.DatetimeIndex(prior[dt])
            if len(prior_times)!=len(times) or prior_times.has_duplicates or not prior_times.sort_values().equals(times):
                raise ValueError(f'Exp17 {name} common Test timestamps do not match reference')
        old=pd.read_csv(previous,encoding='utf-8-sig')
        now=pd.DataFrame(summary)
        keys=['model','scope']
        required=set(keys+['NMAE_pct','n_samples','feature_count'])
        if required.issubset(old.columns):
            merged=now.merge(old[keys+['NMAE_pct','n_samples','feature_count']],on=keys,
                             suffixes=('_Top20','_SFFS'),how='left',validate='one_to_one')
            if merged['NMAE_pct_SFFS'].notna().all():
                if not (merged['n_samples_Top20']==merged['n_samples_SFFS']).all():
                    raise ValueError('Experiment 17 comparison has a different common Test sample count')
                merged['NMAE_improvement_SFFS_vs_Top20_pp']=(
                    merged['NMAE_pct_Top20']-merged['NMAE_pct_SFFS'])
                merged.to_csv(out/'COMPARE_TOP20_vs_SFFS_FIXED_HP.csv',index=False,encoding='utf-8-sig')
                print('\nComparison with exp17 written: COMPARE_TOP20_vs_SFFS_FIXED_HP.csv')
            else:
                print('\n[INFO] Exp17 metrics missing selected model(s); skipped comparison')
    print('\nSaved:',out.resolve())

if __name__=='__main__': main()
