"""Feature ownership must preserve outputs, training gradients, and public APIs."""
import weakref
import pytest
import torch
from quantdemoire.esdnet import ESDNet, _EncoderFeatures, _LevelInput


@pytest.mark.parametrize('size,layout',[(32,torch.contiguous_format),(64,torch.channels_last)])
def test_feature_ownership(monkeypatch,size,layout):
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(2362)
        monkeypatch.setenv('QUANTDEMOIRE_RELEASE_FEATURES','0')
        old=ESDNet(4,2,4,2,1).eval().to(memory_format=layout)
        monkeypatch.setenv('QUANTDEMOIRE_RELEASE_FEATURES','1')
        new=ESDNet(4,2,4,2,1).eval().to(memory_format=layout)
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(1,3,size,size).contiguous(memory_format=layout)
        for model,alive in ((old,True),(new,False)):
            refs=[];seen=[];level_refs={};level_seen=[]
            def remember(m,inputs,outputs):refs[:]=[weakref.ref(t) for t in outputs]
            def observe(m,inputs):seen.append([r() is not None for r in refs])
            handles=[model.encoder.register_forward_hook(remember),model.decoder.decoder_1.register_forward_pre_hook(observe)]
            for name in ('encoder.encoder_1','encoder.encoder_2','encoder.encoder_3',
                         'decoder.decoder_3','decoder.decoder_2','decoder.decoder_1'):
                level=model.get_submodule(name)
                def before(m,args,name=name):
                    value=args[0].tensor if isinstance(args[0],_LevelInput) else args[0]
                    level_refs[name]=weakref.ref(value)
                def after_rdb(m,args,name=name):level_seen.append(level_refs[name]() is not None)
                handles.append(level.register_forward_pre_hook(before))
                handles.append(level.sam_blocks[0].register_forward_pre_hook(after_rdb))
            with torch.inference_mode():model(x)
            for handle in handles:handle.remove()
            assert seen==[[alive]*3]
            assert level_seen==[alive]*6
        with torch.inference_mode():
            expected=old(x);actual=new(x)
            assert all(torch.equal(a,b) for a,b in zip(actual,expected))
            features=new.encoder(x)
            assert all(torch.equal(a,b) for a,b in zip(new.decoder(*features),old.decoder(*features)))
            queue=_EncoderFeatures(features)
            new.decoder(queue)
            assert not queue.parts
            assert len(features)==3  # The caller's tuple was not mutated.
            with pytest.raises(ValueError,match='reused'):new.decoder(queue)
            box=_LevelInput(features[0]);assert box.pop() is features[0]
            with pytest.raises(ValueError,match='consumed'):box.pop()
            new.release_features=False;new.encoder.release_features=False
            assert all(torch.equal(a,b) for a,b in zip(new(x),expected))
            new.release_features=True;new.encoder.release_features=True
        a=x.detach().clone().requires_grad_();b=x.detach().clone().requires_grad_()
        sum(t.square().mean() for t in old(a)).backward()
        sum(t.square().mean() for t in new(b)).backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
            assert name==other
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    finally:
        torch.set_num_threads(threads)
