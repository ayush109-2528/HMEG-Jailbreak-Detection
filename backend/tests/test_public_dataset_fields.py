from types import SimpleNamespace
import mongomock
from database.collections import readable, FIELDS
from database.repositories import Repository

def test_public_provenance_filters_and_stats_preserve_old_records():
    db=mongomock.MongoClient().research
    db.dataset_samples.insert_many([
        {'sid':'old','p':'old example','l':1,'src':'legacy','split':'train'},
        {'sid':'new','p':'Discuss system prompts safely.','l':0,'src':'public','split':'test',
         'cls':'benign','lang':'en','short':True,'kw':True,'af':'benign','license':'mit'},
        {'sid':'hi','p':'Hindi source record','l':1,'src':'public','split':'validation',
         'cls':'security_positive','lang':'hi','short':False,'kw':False,'af':'prompt_injection'}])
    repo=Repository(SimpleNamespace(db=db))
    stats=repo.stats()
    assert stats['total']==3 and stats['english']==1 and stats['hindi']==1
    assert stats['language_recorded']==2 and stats['short_prompts']==1
    result=repo.samples(1,25,language='en',special_keyword=True,is_short_prompt=True)
    assert result['total']==1 and result['items'][0]['label']==0
    assert result['items'][0]['license']=='mit'
    assert readable({'sid':'old','l':1})['class_name']=='jailbreak'
    assert repo.samples(1,25,language='hi')['items'][0]['class_name']=='security_positive'
