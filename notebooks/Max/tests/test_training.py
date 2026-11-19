from test_engine import np, pytest, torch, Net
from models import detach
from train import train_batch
from checkpoint import save,load,rng_state,restore_rng
from data import event_features

def test_resume(tmp_path):
    torch.manual_seed(0); x=torch.randn(2,120,112).half(); y=torch.randn(2,120,2)
    def setup():
        m=Net('lstm',8); opt=torch.optim.Adam(m.parameters(),lr=.002)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=12,pct_start=.3)
        return m,opt,sched
    m,o,s=setup(); train_batch(m,o,s,x,y,40,0)
    save(tmp_path,dict(model=m.state_dict(),opt=o.state_dict(),sched=s.state_dict(),rng=rng_state()))
    train_batch(m,o,s,x,y,40,0)
    n,oo,ss=setup(); ck=load(tmp_path); n.load_state_dict(ck['model']); oo.load_state_dict(ck['opt']); ss.load_state_dict(ck['sched']); restore_rng(ck['rng'])
    train_batch(n,oo,ss,x,y,40,0)
    for p,q in zip(m.parameters(),n.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert ss.state_dict()==s.state_dict()
    pointer=__import__('json').loads((tmp_path/'last.json').read_text()); (tmp_path/pointer['file']).write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'): load(tmp_path)

def test_events_causal():
    x=np.zeros((250,112),np.float32); x[10:,0]=22; x[15:,1]=22
    f=event_features(x); assert f[10,0]==1 and f[15,1]==2
    assert f[60,6]==1 and f[65,6]==0
    np.testing.assert_array_equal(f[:100],event_features(x[:100]))
    assert f[0,5]==np.float32(np.log1p(20000))

def test_causal_model_and_dtype():
    torch.manual_seed(0); m=Net('lstm',8).eval(); x=torch.randn(1,140,112).half()
    with torch.no_grad():
        p,h=m(x); a,h=m(x[:,:70]); b,h=m(x[:,70:],h)
    torch.testing.assert_close(p,torch.cat([a,b],1),atol=1e-6,rtol=1e-5)
    assert p.dtype==torch.float32
