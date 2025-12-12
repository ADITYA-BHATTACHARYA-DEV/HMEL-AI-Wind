# Re-run corrected notebook code (fixing SyntaxError). This will perform the same full demo pipeline as before.
import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Torch for models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Utilities for display in the notebook
from caas_jupyter_tools import display_dataframe_to_user

# ---------------------------------------------------------
# 1) LOAD DATA (try Excel files; if missing, create synthetic sample)
# ---------------------------------------------------------
def try_load_excel(name, sheet_name=0):
    if os.path.exists(name):
        try:
            return pd.read_excel(name, sheet_name=sheet_name)
        except Exception as e:
            print(f"Failed to read {name}: {e}")
            return None
    return None

ws = try_load_excel("windspeeds.xlsx")
rh = try_load_excel("humidity.xlsx")
temp = try_load_excel("temperature.xlsx")
wd = try_load_excel("direction.xlsx")

if ws is None or rh is None or temp is None or wd is None:
    # Build synthetic small dataset using user's example timestamps
    print("Some or all Excel files not found. Using synthetic example data based on user's samples.")
    times = pd.to_datetime([
        "2024-03-10 15:00", "2025-06-24 10:15", "2025-06-24 11:00",
        "2025-06-24 11:10", "2025-06-24 11:20", "2025-06-24 11:30"
    ])
    # Create synthetic repeating measurements for 4 sensors (simulate spatial)
    sensor_ids = ["S1","S2","S3","S4"]
    rows = []
    for s_i, sid in enumerate(sensor_ids):
        for t in times:
            # base wind speed varying by sensor
            base = 1.3 + 0.5*s_i + 0.2*np.sin(t.hour/24*2*np.pi)
            wind = base + np.random.normal(0,0.2)
            rhv = 55 + 5*s_i + np.random.normal(0,1.5)
            tempv = 25 - 0.5*s_i + np.random.normal(0,0.5)
            # wind direction varies per sensor
            wdir = (50 + 30*s_i + 10*np.cos(t.hour/24*2*np.pi) + np.random.normal(0,5)) % 360
            rows.append({"sensor_id":sid,"timestamp":t,"wind_speed":wind,"relative_humidity":rhv,"temperature":tempv,"wind_direction":wdir,"confidence":100,
                         "lat":12.0+0.001*s_i,"lon":76.5+0.001*s_i,"height_m":10 + 5*s_i})
    df = pd.DataFrame(rows)
else:
    # Attempt to unify real data from four sheets
    def normalize_sheet(d, col_prefix):
        d = d.copy()
        ts_cols = [c for c in d.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
        val_cols = [c for c in d.columns if 'value' in c.lower() or 'val' in c.lower()]
        if len(ts_cols)==0:
            raise ValueError("No timestamp column found in sheet")
        if len(val_cols)==0:
            val_cols = [d.select_dtypes(include=np.number).columns[-1]]
        d = d.rename(columns={ts_cols[0]:"timestamp", val_cols[0]:col_prefix})
        return d
    s_ws = normalize_sheet(ws, "wind_speed")
    s_rh = normalize_sheet(rh, "relative_humidity")
    s_temp = normalize_sheet(temp, "temperature")
    s_wd = normalize_sheet(wd, "wind_direction")
    common_cols = set(s_ws.columns) & set(s_rh.columns) & set(s_temp.columns) & set(s_wd.columns)
    key_cols = ["timestamp"]
    if "sensor_id" in common_cols:
        key_cols.append("sensor_id")
    df = s_ws.merge(s_rh, on=key_cols, how='outer').merge(s_temp, on=key_cols, how='outer').merge(s_wd, on=key_cols, how='outer')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df = df.sort_values(['sensor_id','timestamp']).reset_index(drop=True)
print("Data sample:")
display_dataframe_to_user("merged_sample", df.head(12))

# ---------------------------------------------------------
# 2) PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])
out_frames = []
for sid, g in df.groupby('sensor_id'):
    g = g.set_index('timestamp').sort_index()
    g = g.resample('10T').mean()
    for c in ['lat','lon','height_m']:
        if c in df.columns:
            g[c] = g[c].fillna(method='ffill').fillna(method='bfill')
    g['sensor_id'] = sid
    for c in ['wind_speed','relative_humidity','temperature','wind_direction','confidence']:
        if c in g.columns:
            g[c] = g[c].interpolate(limit=3).fillna(method='ffill').fillna(method='bfill')
    out_frames.append(g.reset_index())
df_reg = pd.concat(out_frames).reset_index(drop=True)
df_reg['wind_dir_rad'] = np.deg2rad(df_reg['wind_direction'] % 360)
df_reg['dir_x'] = np.cos(df_reg['wind_dir_rad'])
df_reg['dir_y'] = np.sin(df_reg['wind_dir_rad'])
if 'wind_speed' in df_reg.columns:
    df_reg['u'] = df_reg['wind_speed'] * df_reg['dir_x']
    df_reg['v'] = df_reg['wind_speed'] * df_reg['dir_y']

df_reg['hour'] = df_reg['timestamp'].dt.hour
df_reg['dow'] = df_reg['timestamp'].dt.dayofweek
df_reg['month'] = df_reg['timestamp'].dt.month
df_reg['hour_sin'] = np.sin(2*np.pi*df_reg['hour']/24)
df_reg['hour_cos'] = np.cos(2*np.pi*df_reg['hour']/24)

print("Preprocessed sample:")
display_dataframe_to_user("preprocessed_sample", df_reg.head(12))

# ---------------------------------------------------------
# 3) Prepare dataset for LSTM/BiLSTM forecasting of wind_speed (one-step ahead)
# ---------------------------------------------------------
seq_len = 6
pred_steps = 1
features = ['wind_speed','u','v','relative_humidity','temperature','hour_sin','hour_cos']
df_reg[features] = df_reg[features].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
sensors = df_reg['sensor_id'].unique().tolist()
sensor_to_idx = {s:i for i,s in enumerate(sensors)}
df_reg['sensor_idx'] = df_reg['sensor_id'].map(sensor_to_idx)
num_sensors = len(sensors)

class SeqDataset(Dataset):
    def __init__(self, df, seq_len=6, pred_steps=1):
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.X = []
        self.Y = []
        for sid, g in df.groupby('sensor_idx'):
            arr = g.sort_values('timestamp')[features].values
            for i in range(len(arr)-seq_len-pred_steps+1):
                seq = arr[i:i+seq_len]
                targ = arr[i+seq_len:i+seq_len+pred_steps][:,0]
                self.X.append(seq)
                self.Y.append(targ)
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.Y[idx]

dataset = SeqDataset(df_reg, seq_len=seq_len, pred_steps=pred_steps)
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

print(f"Sequences prepared: total={len(dataset)}, train={train_size}, val={val_size}")

# ---------------------------------------------------------
# 4) Define LSTM and BiLSTM models (PyTorch)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

class BiLSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

input_size = len(features)

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    history = {'train_loss':[], 'val_loss':[]}
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset) if len(train_loader.dataset)>0 else 1.0
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device).squeeze(-1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vl += loss.item() * xb.size(0)
        vl /= len(val_loader.dataset) if len(val_loader.dataset)>0 else 1.0
        history['train_loss'].append(tl); history['val_loss'].append(vl)
    return model, history

lstm = LSTMForecast(input_size=input_size, hidden_size=32, num_layers=1)
bilstm = BiLSTMForecast(input_size=input_size, hidden_size=32, num_layers=1)

lstm, hist_l = train_model(lstm, train_loader, val_loader, epochs=20, lr=1e-3)
bilstm, hist_b = train_model(bilstm, train_loader, val_loader, epochs=20, lr=1e-3)

plt.figure(figsize=(8,4))
plt.plot(hist_l['train_loss'], label='LSTM train'); plt.plot(hist_l['val_loss'], label='LSTM val')
plt.plot(hist_b['train_loss'], '--', label='BiLSTM train'); plt.plot(hist_b['val_loss'], '--', label='BiLSTM val')
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend(); plt.title("Training history (demo)")
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/training_history.png")
plt.show()

# ---------------------------------------------------------
# 5) Make predictions on last sequence per sensor and show example plot
# ---------------------------------------------------------
def predict_last(model, df_reg, features, seq_len=seq_len):
    model = model.to(device); model.eval()
    preds = []
    for sid, g in df_reg.groupby('sensor_id'):
        g = g.sort_values('timestamp')
        if len(g) < seq_len: continue
        seq = g[features].values[-seq_len:][None,:,:].astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(seq).to(device)).cpu().numpy().ravel()[0]
        preds.append({"sensor_id":sid, "timestamp":g['timestamp'].iloc[-1]+pd.Timedelta(minutes=10), "pred_wind_speed":float(out),
                      "last_observed": float(g['wind_speed'].values[-1])})
    return pd.DataFrame(preds)

pred_df_lstm = predict_last(lstm, df_reg, features, seq_len=seq_len)
pred_df_bilstm = predict_last(bilstm, df_reg, features, seq_len=seq_len)

print("Sample predictions (LSTM):")
display_dataframe_to_user("predictions_lstm", pred_df_lstm)
print("Sample predictions (BiLSTM):")
display_dataframe_to_user("predictions_bilstm", pred_df_bilstm)

sensor_example = sensors[0]
g = df_reg[df_reg['sensor_id']==sensor_example].sort_values('timestamp')
plt.figure(figsize=(8,3))
plt.plot(g['timestamp'], g['wind_speed'], marker='o', label='observed (last values)')
if sensor_example in pred_df_lstm['sensor_id'].values:
    p = pred_df_lstm[pred_df_lstm['sensor_id']==sensor_example].iloc[0]
    plt.scatter([p['timestamp']], [p['pred_wind_speed']], color='red', label='LSTM pred')
if sensor_example in pred_df_bilstm['sensor_id'].values:
    p2 = pred_df_bilstm[pred_df_bilstm['sensor_id']==sensor_example].iloc[0]
    plt.scatter([p2['timestamp']], [p2['pred_wind_speed']], color='green', label='BiLSTM pred')
plt.xlabel("Time"); plt.ylabel("Wind speed (m/s)"); plt.title(f"Observed & next-step predictions for {sensor_example}"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/prediction_example.png")
plt.show()

# ---------------------------------------------------------
# 6) Simple GNN-like fusion across sensors (demo)
# ---------------------------------------------------------
node_feats = []
for sid, g in df_reg.groupby('sensor_id'):
    recent = g.sort_values('timestamp').tail(6)
    u_mean = recent['u'].mean(); v_mean = recent['v'].mean()
    dir_var = recent['wind_dir_rad'].std()
    node_feats.append([u_mean, v_mean, dir_var])
node_feats = np.array(node_feats, dtype=np.float32)
coords = []
for s in sensors:
    sub = df_reg[df_reg['sensor_id']==s].iloc[0]
    coords.append((sub['lat'], sub['lon']))
coords = np.array(coords)
dmat = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2))
adj = 1/(dmat+1e-6)
np.fill_diagonal(adj, 0.0)
adj_norm = adj / (adj.sum(axis=1, keepdims=True)+1e-6)

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
    def forward(self, X, A_norm):
        XW = self.W(X)
        out = torch.matmul(torch.tensor(A_norm, dtype=torch.float32, device=XW.device), XW)
        return torch.relu(out)

gcn = SimpleGCN(in_dim=node_feats.shape[1], out_dim=4).to(device)
X = torch.tensor(node_feats, dtype=torch.float32).to(device)
A_norm = torch.tensor(adj_norm, dtype=torch.float32).to(device)
with torch.no_grad():
    node_emb = gcn(X, A_norm).cpu().numpy()

print("GCN node embeddings (demo):")
print(node_emb)

# ---------------------------------------------------------
# 7) PINN-like vertical profile fit (corrected)
# ---------------------------------------------------------
sensor0 = df_reg[df_reg['sensor_id']==sensors[0]].iloc[-1]
Uref = sensor0['wind_speed']
zref = sensor0['height_m']
alpha = 0.22
zs = np.array([5,10,20,30,40,50], dtype=np.float32)
U_true = Uref * (zs / zref)**alpha + np.random.normal(0,0.05,len(zs))

class PINNProfile(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh(), nn.Linear(32,1))
    def forward(self,z):
        return self.net(z.view(-1,1)).squeeze(-1)

pinn = PINNProfile().to(device)
opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
z0 = 0.3
kappa = 0.4
zs_t = torch.tensor(zs, dtype=torch.float32, device=device)
U_true_t = torch.tensor(U_true, dtype=torch.float32, device=device)
for epoch in range(1000):
    pinn.train()
    pred = pinn(zs_t)
    data_loss = nn.MSELoss()(pred, U_true_t)
    lnz = torch.log(zs_t/(z0+1e-6))
    # closed-form scale factor c
    c = (pred.detach()*lnz).sum() / (lnz*lnz).sum()
    phys_target = c * lnz
    phys_loss = nn.MSELoss()(pred, phys_target)
    loss = data_loss + 0.5*phys_loss
    opt.zero_grad(); loss.backward(); opt.step()

pinn.eval()
with torch.no_grad():
    zs_plot = torch.tensor(np.linspace(5,60,50,dtype=np.float32), device=device)
    U_pred = pinn(zs_plot).cpu().numpy()

plt.figure(figsize=(4,5))
plt.plot(U_true, zs, 'o', label='synthetic measures')
plt.plot(U_pred, zs_plot.cpu().numpy(), '-', label='PINN prediction')
plt.gca().invert_yaxis()
plt.xlabel("U (m/s)"); plt.ylabel("Height (m)"); plt.title("Vertical profile (PINN demo)"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/vertical_profile_pinn.png")
plt.show()

# ---------------------------------------------------------
# 8) Candidate site scoring & GeoJSON export for Leaflet
# ---------------------------------------------------------
latmin, lonmin = coords.min(axis=0)
latmax, lonmax = coords.max(axis=0)
n_x, n_y = 6,6
lats = np.linspace(latmin-0.001, latmax+0.001, n_x)
lons = np.linspace(lonmin-0.001, lonmax+0.001, n_y)

candidates = []
for lat in lats:
    for lon in lons:
        dists = np.sqrt(((coords - np.array([lat, lon]))**2).sum(axis=1))
        nearest = dists.argmin()
        mean_ws = df_reg[df_reg['sensor_id']==sensors[nearest]]['wind_speed'].mean()
        energy_proxy = mean_ws**3
        penalty = 0.0
        score = energy_proxy * (1 - penalty)
        candidates.append({"lat":float(lat),"lon":float(lon),"score":float(score)})
cand_df = pd.DataFrame(candidates).sort_values('score',ascending=False).reset_index(drop=True)
display_dataframe_to_user("candidate_scores", cand_df.head(10))

geo = {"type":"FeatureCollection","features":[]}
for i,row in cand_df.iterrows():
    feat = {"type":"Feature","geometry":{"type":"Point","coordinates":[row['lon'], row['lat']]},
            "properties":{"score":row['score'], "rank":int(i+1)}}
    geo['features'].append(feat)
with open("/mnt/data/candidate_sites.geojson","w") as f:
    json.dump(geo, f)

# ---------------------------------------------------------
# 9) Save key outputs and provide summary tables / files
# ---------------------------------------------------------
pred_df_lstm.to_csv("/mnt/data/predictions_lstm.csv", index=False)
pred_df_bilstm.to_csv("/mnt/data/predictions_bilstm.csv", index=False)
cand_df.to_csv("/mnt/data/candidate_scores.csv", index=False)

print("Saved files: /mnt/data/training_history.png, /mnt/data/prediction_example.png, /mnt/data/vertical_profile_pinn.png")
print("Saved CSVs: /mnt/data/predictions_lstm.csv, /mnt/data/predictions_bilstm.csv, /mnt/data/candidate_scores.csv")
print("Saved GeoJSON: /mnt/data/candidate_sites.geojson")

summary = pd.DataFrame([
    {"artifact":"training_history.png","path":"/mnt/data/training_history.png"},
    {"artifact":"prediction_example.png","path":"/mnt/data/prediction_example.png"},
    {"artifact":"vertical_profile_pinn.png","path":"/mnt/data/vertical_profile_pinn.png"},
    {"artifact":"predictions_lstm.csv","path":"/mnt/data/predictions_lstm.csv"},
    {"artifact":"predictions_bilstm.csv","path":"/mnt/data/predictions_bilstm.csv"},
    {"artifact":"candidate_sites.geojson","path":"/mnt/data/candidate_sites.geojson"}
])
display_dataframe_to_user("artifacts", summary)

print("Demo complete. Please replace synthetic data with your Excel sheets and re-run for real outputs.")
