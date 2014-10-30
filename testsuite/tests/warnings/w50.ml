
let rec fact = function
  | 1 -> 1
  | n -> n * [%tailcall fact (n-1)]
;;
