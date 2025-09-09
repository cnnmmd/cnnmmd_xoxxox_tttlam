import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from xoxxox.shared import Custom

#---------------------------------------------------------------------------

class TttPrc:

  def __init__(self, config="xoxxox/config_tttlam_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    nmodel = diccnf["nmodel"]
    cnfbnb = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16)
    self.omodel = AutoModelForCausalLM.from_pretrained(
      nmodel,
      device_map="auto",
      quantization_config=cnfbnb)
    self.tokenz = AutoTokenizer.from_pretrained(
      nmodel)

  def status(self, config="xoxxox/config_tttlam_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.nummax = diccnf["prmmax"]
    self.prmsys = diccnf["status"]
    self.numout = diccnf["numout"]
    self.numtmp = diccnf["numtmp"]
    self.prompt = ""
    self.lstusr = []
    self.lstres = []

  def addusr(self, prmusr, lstusr):
    lstusr.append(prmusr)
    if len(lstusr) > self.nummax:
      lstusr.pop(0)
    return lstusr

  def addres(self, outres, lstres):
    lstres.append(outres)
    if len(lstres) > self.nummax - 1:
      lstres.pop(0)
    return lstres

  def genprm(self, lstusr, lstres):
    prompt = ""
    i = 0
    while i < len(lstusr):
      if i == 0:
        if i == len(lstusr) - 1:
          prompt = "<s>" + "[INST]" + "<<SYS>>" + self.prmsys + "<</SYS>>" + lstusr[i] + "[/INST]"
        else:
          prompt = "<s>" + "[INST]" + "<<SYS>>" + self.prmsys + "<</SYS>>" + lstusr[i] + "[/INST]" + lstres[i]
      else:
        if i == len(lstusr) - 1:
          prompt = prompt + "</s>" + "<s>" + "[INST]" + lstusr[i] + "[/INST]"
        else:
          prompt = prompt + "</s>" + "<s>" + "[INST]" + lstusr[i] + "[/INST]" +  lstres[i]
      i = i + 1
    return prompt

  def infere(self, prmusr):
    self.addusr(prmusr, self.lstusr)
    self.prompt = self.genprm(self.lstusr, self.lstres)

    inputs = self.tokenz([self.prompt], return_tensors="pt").to("cuda")
    lstgen = self.omodel.generate(
      **inputs,
      max_new_tokens=self.numout,
      temperature=self.numtmp,
      do_sample=True,
      pad_token_id=self.tokenz.eos_token_id)
    output = self.tokenz.decode(lstgen[0]) # 推定結果の全体
    outres = self.tokenz.decode(lstgen[0][len(inputs["input_ids"][0]):]) # 推定結果の追加部分（応答）

    outres = outres[:outres.find("<")]
    outres = outres[:outres.find("[")]
    outres = outres.replace(" ", "")
    outres = outres.replace("\n", "")
    outres = outres.replace("「", "")
    outres = outres.replace("」", "")

    #print("prompt[" + self.prompt + "]") # DBG
    #print("output[" + output + "]") # DBG
    #print("outres[" + outres + "]") # DBG

    self.addres(outres, self.lstres)

    return (outres, "")
