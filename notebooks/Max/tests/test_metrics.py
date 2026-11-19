from test_engine import np,pytest
from metrics import sums,corr

def test_warmup_mask_clip():
    y=np.ones((200,2)); y[99:150]*=-1; p=y.copy(); m=np.ones(200,bool)
    p[:99]=100
    np.testing.assert_allclose(corr(sums(p,y,m)),[1,1])
    p[150:]=100; np.testing.assert_allclose(sums(p,y,m),sums(np.clip(p,-2,2),y,m))
    m[:]=False
    with pytest.raises(ValueError): corr(sums(p,y,m))
