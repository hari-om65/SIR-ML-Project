with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'r') as f:
    code = f.read()

# Fix 1: Update SIRMLP to ImprovedSIRMLP with new architecture
old_arch = '''class SIRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,128), nn.Tanh(),
            nn.Linear(128,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)'''

new_arch = '''class SIRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),   nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)'''

code = code.replace(old_arch, new_arch)

# Fix 2: Update SIR_MCDropout to match new architecture
old_mc = '''class SIR_MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128,256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128,3),
        )'''

new_mc = '''class SIR_MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),   nn.Tanh(), nn.Dropout(p),
            nn.Linear(256, 512), nn.Tanh(), nn.Dropout(p),
            nn.Linear(512, 512), nn.Tanh(), nn.Dropout(p),
            nn.Linear(512, 256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256, 128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128, 3),
        )'''

code = code.replace(old_mc, new_mc)

# Fix 3: Add input normalization everywhere inp_t is built
# Replace all occurrences of [[b, g, t/t_max] with normalized version
old_inp1 = "inp_t = torch.tensor(\n        [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)"
new_inp1 = "inp_t = torch.tensor(\n        [[b/0.9, g/0.5, t/t_max] for t in t_grid], dtype=torch.float32)"
code = code.replace(old_inp1, new_inp1)

old_inp2 = "inp_t  = torch.tensor(\n        [[b, g, t/t_max] for t in t_test], dtype=torch.float32,"
new_inp2 = "inp_t  = torch.tensor(\n        [[b/0.9, g/0.5, t/t_max] for t in t_test], dtype=torch.float32,"
code = code.replace(old_inp2, new_inp2)

old_inp3 = "inp_t = torch.tensor([[b_est,g_est,t/t_max]\n                          for t in t_grid], dtype=torch.float32)"
new_inp3 = "inp_t = torch.tensor([[b_est/0.9, g_est/0.5, t/t_max]\n                          for t in t_grid], dtype=torch.float32)"
code = code.replace(old_inp3, new_inp3)

# Fix 4: Normalize inside solve_inverse gradient loop
old_inv = "        inp = torch.stack([bc.expand(len(t_grid)),\n                           gc.expand(len(t_grid)), t_n], dim=1)"
new_inv = "        inp = torch.stack([bc.expand(len(t_grid))/0.9,\n                           gc.expand(len(t_grid))/0.5, t_n], dim=1)"
code = code.replace(old_inv, new_inv)

# Fix 5: Normalize inside run_robustness
old_rob = "            inp_t = torch.tensor(\n                [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)"
new_rob = "            inp_t = torch.tensor(\n                [[b/0.9, g/0.5, t/t_max] for t in t_grid], dtype=torch.float32)"
code = code.replace(old_rob, new_rob)

with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'w') as f:
    f.write(code)

print("Patch 6 (new architecture + input normalization) applied!")
