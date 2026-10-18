"""Independent, matched-budget optimizer experiment; no test-set evaluation.

Run from the repository: python -m Utils.optimizer_audit
Or call run_optimizer_audit using the notebook's preserved training/validation arrays.
Original checkpoints, CSVs and final-model registration records are never replaced.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def build_audit_model(balanced=False):
    layers=[keras.Input(shape=(64,64,3)),keras.layers.Rescaling(1./255)]
    for index,filters in enumerate([32,64,128,256]):
        options={}
        if balanced:
            # Each depthwise output sums nine inputs, not nine times Cin.
            options={'depthwise_initializer':keras.initializers.RandomNormal(stddev=1./3),
                     'pointwise_initializer':keras.initializers.HeNormal()}
        layers.append(keras.layers.SeparableConv2D(filters,3,activation='relu',
                      padding='same' if index==0 else 'valid',**options))
        if index<3: layers.append(keras.layers.MaxPooling2D((2,2)))
    layers.extend([keras.layers.GlobalAveragePooling2D(),keras.layers.Dropout(.4),
                   keras.layers.Dense(6,activation='softmax')])
    return keras.Sequential(layers,name='model_b_optimizer_audit')

def fingerprint(x,y,labels):
    d=hashlib.sha256()
    d.update(json.dumps(np.asarray(labels).tolist()).encode())
    d.update(str(x.shape).encode()); d.update(str(x.dtype).encode()); d.update(y.tobytes())
    for image in x: d.update(image.tobytes())
    return d.hexdigest()

def initialization_probe(x,y,balanced):
    keras.utils.set_random_seed(42)
    model=build_audit_model(balanced)
    xb=tf.convert_to_tensor(x[:64]); yb=tf.convert_to_tensor(y[:64])
    with tf.GradientTape() as tape:
        probabilities=model(xb,training=False)
        loss=keras.losses.sparse_categorical_crossentropy(yb,probabilities)
        loss=tf.reduce_mean(loss)
    gradients=tape.gradient(loss,model.trainable_weights)
    rows=[]
    for weight,gradient in zip(model.trainable_weights,gradients):
        if gradient is not None:
            rows.append({'weight':getattr(weight,'path',weight.name),
                         'gradient_l2':float(tf.linalg.norm(gradient).numpy())})
    kernel_norms=[r['gradient_l2'] for r in rows if 'kernel' in r['weight']]
    return {'initialization':'variance_preserving' if balanced else 'default_glorot',
            'loss':float(loss.numpy()),'kernel_gradient_median':float(np.median(kernel_norms)),
            'gradients':rows}

class AuditTimer(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs=None): self.start=time.perf_counter()
    def on_epoch_end(self,epoch,logs=None):
        logs['epoch_seconds']=time.perf_counter()-self.start

def run_optimizer_audit(train_x,train_y,val_x,val_y,labels,repo_root=None,epochs=20):
    root=Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    assert epochs==20 and len(train_y)==11923 and len(val_y)==2555
    run=root/'sota_results/optimizer_audit'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True,exist_ok=False)
    probes=[initialization_probe(train_x,train_y,b) for b in [False,True]]
    # Set before training, using training-batch gradients only; no validation/test tuning.
    balanced=probes[0]['kernel_gradient_median'] < probes[1]['kernel_gradient_median']/10
    config={'epochs':epochs,'batch_size':64,'seed':42,'initialization':
            'variance_preserving_depthwise_pointwise' if balanced else 'default_glorot',
            'initialization_selection':'training-batch kernel gradient median, before optimizer trials',
            'depthwise_stddev':1./3 if balanced else None,
            'pointwise_initializer':'HeNormal' if balanced else 'GlorotUniform',
            'labels':np.asarray(labels).tolist(),
            'split_fingerprints':{'train':fingerprint(train_x,train_y,labels),
                                  'validation':fingerprint(val_x,val_y,labels)},
            'tensorflow':tf.__version__,'initialization_probes':probes,
            'test_data_used':False,
            'optimizers':[{'name':'Adam','learning_rate':.0005,'momentum':None},
                          {'name':'SGD','learning_rate':.05,'momentum':0.},
                          {'name':'SGD + momentum','learning_rate':.05,'momentum':.9}]}
    keras.utils.set_random_seed(42)
    initial=build_audit_model(balanced); initial_weights=initial.get_weights()
    config['initial_weights_sha256']=hashlib.sha256(b''.join(w.tobytes() for w in initial_weights)).hexdigest()
    (run/'config.json').write_text(json.dumps(config,indent=2))
    print('RUN:',run,flush=True)
    print('INITIALIZATION PROBES:',json.dumps(probes),flush=True)
    print('SELECTED INITIALIZATION:',config['initialization'],flush=True)
    records=[]
    for spec in config['optimizers']:
        keras.backend.clear_session(); keras.utils.set_random_seed(42)
        model=build_audit_model(balanced); model.set_weights(initial_weights)
        optimizer=keras.optimizers.Adam(spec['learning_rate']) if spec['name']=='Adam' else keras.optimizers.SGD(spec['learning_rate'],momentum=spec['momentum'])
        model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        folder=run/('adam' if spec['name']=='Adam' else 'sgd' if spec['momentum']==0 else 'sgd_momentum')
        folder.mkdir()
        print('\nTRAINING',spec['name'],flush=True)
        history=model.fit(train_x,train_y,validation_data=(val_x,val_y),epochs=epochs,
                          batch_size=64,shuffle=True,verbose=2,
                          callbacks=[AuditTimer(),keras.callbacks.ModelCheckpoint(str(folder/'best.keras'),
                                     monitor='val_accuracy',mode='max',save_best_only=True),
                                     keras.callbacks.CSVLogger(str(folder/'history.csv')),
                                     keras.callbacks.TerminateOnNaN()])
        frame=pd.read_csv(folder/'history.csv'); assert len(frame)==epochs
        selected=int(frame['val_accuracy'].argmax()); row=frame.iloc[selected]
        best=keras.models.load_model(folder/'best.keras',compile=False)
        best.save(folder/'model.keras')
        record={**spec,'best_epoch':selected+1,'best_validation_accuracy':float(row['val_accuracy']),
                'validation_loss_at_best_epoch':float(row['val_loss']),
                'final_training_accuracy':float(frame.iloc[-1]['accuracy']),
                'final_validation_accuracy':float(frame.iloc[-1]['val_accuracy']),
                'mean_seconds_per_epoch':float(frame['epoch_seconds'].mean()),
                'epochs_completed':len(frame),'directory':folder.relative_to(root).as_posix(),
                'checkpoint_sha256':hashlib.sha256((folder/'model.keras').read_bytes()).hexdigest()}
        records.append(record)
        pd.DataFrame(records).to_csv(run/'optimizer_results.csv',index=False)
    info={'training_completed':True,'run_directory':run.relative_to(root).as_posix(),
          'config':config,'results':records}
    (run/'completed.json').write_text(json.dumps(info,indent=2))
    (run.parent/'latest_completed.json').write_text(json.dumps({'run_directory':info['run_directory']},indent=2))
    print('COMPLETED:',json.dumps(info),flush=True)
    return info

def main():
    from sklearn.model_selection import train_test_split
    from Utils.utils import loadData
    root=Path(__file__).resolve().parents[1]
    a,ay,labels=loadData('train'); b,by,_=loadData('test',labels)
    x=np.concatenate([a,b]); y=np.concatenate([ay,by]); del a,b,ay,by
    tx,remaining,ty,ry=train_test_split(x,y,test_size=.30,random_state=42,stratify=y)
    vx,unused_test,vy,unused_test_y=train_test_split(remaining,ry,test_size=.50,random_state=42,stratify=ry)
    del x,y,remaining,ry,unused_test,unused_test_y
    run_optimizer_audit(tx,ty,vx,vy,labels,root)

if __name__=='__main__': main()
