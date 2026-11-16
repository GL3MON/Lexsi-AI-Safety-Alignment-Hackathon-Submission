"""Ground calculator operands in actual observations and verify financial answers."""
import ast,re
from decimal import Decimal,InvalidOperation

def number(value):return Decimal(str(value).replace(',','')).quantize(Decimal('.01'))
def prices(value):
 if isinstance(value,dict):
  for k,v in value.items():
   if k in ('price','amount','price_difference') and isinstance(v,(int,float,str)):
    try:yield abs(number(v))
    except InvalidOperation:pass
   else:yield from prices(v)
 elif isinstance(value,list):
  for v in value:yield from prices(v)
def validate_calculation(expression,audit,budget=None,visible=''):
 tree=ast.parse(expression,mode='eval')
 if not any(isinstance(n,ast.BinOp) for n in ast.walk(tree)):raise ValueError('calculator_must_perform_arithmetic')
 allowed_nodes=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd,ast.Constant)
 if any(not isinstance(n,allowed_nodes) for n in ast.walk(tree)):raise ValueError('unsupported_calculator_expression')
 allowed=set()
 for event in audit:
  name=event['call']['name'];result=event['result']
  if name in ('get_order_details','get_product_details'):allowed.update(prices(result))
  elif name=='calculate':
   try:allowed.add(abs(number(result)))
   except (InvalidOperation,TypeError):pass
 if budget is not None and number(budget) in {number(x) for x in re.findall(r'(?<!\w)\d+(?:\.\d+)?',visible.replace(',',''))}:allowed.add(abs(number(budget)))
 constants=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)]
 if any(type(v) not in (int,float) or abs(number(v)) not in allowed for v in constants):raise ValueError('calculator_operand_not_from_observed_prices_or_customer_budget')
 return True

def calculation_satisfied(goal,audit):
 for event in audit:
  if event['call']['name']!='calculate':continue
  try:
   actual=number(event['result']);expected=number(goal['expected'])
   if (abs(actual)==abs(expected) if goal['kind']!='refund_total' else actual==expected):return True
  except (InvalidOperation,TypeError):pass
 return False

def answer_matches(goal,text):
 amounts={number(x) for x in re.findall(r'(?<!\w)\d+(?:\.\d+)?',text.replace(',',''))}
 if abs(number(goal['expected'])) not in amounts:return False
 if goal['kind']=='budget':
  pattern=r'over|exceed|short|not enough|not fit' if number(goal['expected'])<0 else r'within|under|left|remain|fits|enough'
  return bool(re.search(pattern,text,re.I))
 return True
