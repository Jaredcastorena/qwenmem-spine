import Base

type Side is Data:
  Left{}
  Right{}

type Triangle is Data:
  Branch{apex: U32, left: U32, right: U32}

def preferred(t: Triangle, side: Side) -> U32:
  match t:
    case Branch{apex, left, right}:
      match side:
        case Left{}: left
        case Right{}: right

def alternate(t: Triangle, side: Side) -> U32:
  match t:
    case Branch{apex, left, right}:
      match side:
        case Left{}: right
        case Right{}: left

def apex(t: Triangle) -> U32:
  match t:
    case Branch{apex, left, right}: apex

