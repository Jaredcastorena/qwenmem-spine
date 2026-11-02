import Base
import ./navigation.bend as N

law preferred_left:
  for a: U32
  for l: U32
  for r: U32
  {N.preferred(N.Branch{a,l,r}, N.Left{}) == l : U32}

law preferred_right:
  for a: U32
  for l: U32
  for r: U32
  {N.preferred(N.Branch{a,l,r}, N.Right{}) == r : U32}

law alternate_left:
  for a: U32
  for l: U32
  for r: U32
  {N.alternate(N.Branch{a,l,r}, N.Left{}) == r : U32}

law alternate_right:
  for a: U32
  for l: U32
  for r: U32
  {N.alternate(N.Branch{a,l,r}, N.Right{}) == l : U32}

law apex_preserved:
  for a: U32
  for l: U32
  for r: U32
  {N.apex(N.Branch{a,l,r}) == a : U32}

