(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* Exceptions *)

external register_named_value : string -> 'a -> unit
                              = "caml_register_named_value"

let () =
  (* for runtime/fail_nat.c *)
  register_named_value "Pervasives.array_bound_error"
    (Invalid_argument "index out of bounds")

external raise : exn -> 'a = "%raise"
external raise_notrace : exn -> 'a = "%raise_notrace"

let failwith s = raise(Failure s)
let invalid_arg s = raise(Invalid_argument s)

exception Exit
exception Match_failure = Match_failure
exception Assert_failure = Assert_failure
exception Invalid_argument = Invalid_argument
exception Failure = Failure
exception Not_found = Not_found
exception Out_of_memory = Out_of_memory
exception Stack_overflow = Stack_overflow
exception Sys_error = Sys_error
exception End_of_file = End_of_file
exception Division_by_zero = Division_by_zero
exception Sys_blocked_io = Sys_blocked_io
exception Undefined_recursive_module = Undefined_recursive_module

(* Composition operators *)

external ( |> ) : 'a -> ('a -> 'b) -> 'b = "%revapply"
external ( @@ ) : ('a -> 'b) -> 'a -> 'b = "%apply"

(* Debugging *)

external __LOC__ : string = "%loc_LOC"
external __FILE__ : string = "%loc_FILE"
external __LINE__ : int = "%loc_LINE"
external __MODULE__ : string = "%loc_MODULE"
external __POS__ : string * int * int * int = "%loc_POS"
external __FUNCTION__ : string = "%loc_FUNCTION"

external __LOC_OF__ : 'a -> string * 'a = "%loc_LOC"
external __LINE_OF__ : 'a -> int * 'a = "%loc_LINE"
external __POS_OF__ : 'a -> (string * int * int * int) * 'a = "%loc_POS"

(* Comparisons *)

external ( = ) : 'a -> 'a -> bool = "%equal"
external ( <> ) : 'a -> 'a -> bool = "%notequal"
external ( < ) : 'a -> 'a -> bool = "%lessthan"
external ( > ) : 'a -> 'a -> bool = "%greaterthan"
external ( <= ) : 'a -> 'a -> bool = "%lessequal"
external ( >= ) : 'a -> 'a -> bool = "%greaterequal"
external compare : 'a -> 'a -> int = "%compare"

let min x y = if x <= y then x else y
let max x y = if x >= y then x else y

external ( == ) : 'a -> 'a -> bool = "%eq"
external ( != ) : 'a -> 'a -> bool = "%noteq"

(* Boolean operations *)

external not : bool -> bool = "%boolnot"
external ( & ) : bool -> bool -> bool = "%sequand"
external ( && ) : bool -> bool -> bool = "%sequand"
external ( or ) : bool -> bool -> bool = "%sequor"
external ( || ) : bool -> bool -> bool = "%sequor"

(* Integer operations *)

external ( ~- ) : int -> int = "%negint"
external ( ~+ ) : int -> int = "%identity"
external succ : int -> int = "%succint"
external pred : int -> int = "%predint"
external ( + ) : int -> int -> int = "%addint"
external ( - ) : int -> int -> int = "%subint"
external ( * ) : int -> int -> int = "%mulint"
external ( / ) : int -> int -> int = "%divint"
external ( mod ) : int -> int -> int = "%modint"

let abs x = if x >= 0 then x else -x

external ( land ) : int -> int -> int = "%andint"
external ( lor ) : int -> int -> int = "%orint"
external ( lxor ) : int -> int -> int = "%xorint"

let lnot x = x lxor (-1)

external ( lsl ) : int -> int -> int = "%lslint"
external ( lsr ) : int -> int -> int = "%lsrint"
external ( asr ) : int -> int -> int = "%asrint"

let max_int = (-1) lsr 1
let min_int = max_int + 1

(* Floating-point operations *)

external ( ~-. ) : float -> float = "%negfloat"
external ( ~+. ) : float -> float = "%identity"
external ( +. ) : float -> float -> float = "%addfloat"
external ( -. ) : float -> float -> float = "%subfloat"
external ( *. ) : float -> float -> float = "%mulfloat"
external ( /. ) : float -> float -> float = "%divfloat"
external ( ** ) : float -> float -> float = "caml_power_float" "pow"
  [@@unboxed] [@@noalloc]
external exp : float -> float = "caml_exp_float" "exp" [@@unboxed] [@@noalloc]
external expm1 : float -> float = "caml_expm1_float" "caml_expm1"
  [@@unboxed] [@@noalloc]
external acos : float -> float = "caml_acos_float" "acos"
  [@@unboxed] [@@noalloc]
external asin : float -> float = "caml_asin_float" "asin"
  [@@unboxed] [@@noalloc]
external atan : float -> float = "caml_atan_float" "atan"
  [@@unboxed] [@@noalloc]
external atan2 : float -> float -> float = "caml_atan2_float" "atan2"
  [@@unboxed] [@@noalloc]
external hypot : float -> float -> float
               = "caml_hypot_float" "caml_hypot" [@@unboxed] [@@noalloc]
external cos : float -> float = "caml_cos_float" "cos" [@@unboxed] [@@noalloc]
external cosh : float -> float = "caml_cosh_float" "cosh"
  [@@unboxed] [@@noalloc]
external log : float -> float = "caml_log_float" "log" [@@unboxed] [@@noalloc]
external log10 : float -> float = "caml_log10_float" "log10"
  [@@unboxed] [@@noalloc]
external log1p : float -> float = "caml_log1p_float" "caml_log1p"
  [@@unboxed] [@@noalloc]
external sin : float -> float = "caml_sin_float" "sin" [@@unboxed] [@@noalloc]
external sinh : float -> float = "caml_sinh_float" "sinh"
  [@@unboxed] [@@noalloc]
external sqrt : float -> float = "caml_sqrt_float" "sqrt"
  [@@unboxed] [@@noalloc]
external tan : float -> float = "caml_tan_float" "tan" [@@unboxed] [@@noalloc]
external tanh : float -> float = "caml_tanh_float" "tanh"
  [@@unboxed] [@@noalloc]
external ceil : float -> float = "caml_ceil_float" "ceil"
  [@@unboxed] [@@noalloc]
external floor : float -> float = "caml_floor_float" "floor"
  [@@unboxed] [@@noalloc]
external abs_float : float -> float = "%absfloat"
external copysign : float -> float -> float
                  = "caml_copysign_float" "caml_copysign"
                  [@@unboxed] [@@noalloc]
external mod_float : float -> float -> float = "caml_fmod_float" "fmod"
  [@@unboxed] [@@noalloc]
external frexp : float -> float * int = "caml_frexp_float"
external ldexp : (float [@unboxed]) -> (int [@untagged]) -> (float [@unboxed]) =
  "caml_ldexp_float" "caml_ldexp_float_unboxed" [@@noalloc]
external modf : float -> float * float = "caml_modf_float"
external float : int -> float = "%floatofint"
external float_of_int : int -> float = "%floatofint"
external truncate : float -> int = "%intoffloat"
external int_of_float : float -> int = "%intoffloat"
external float_of_bits : int64 -> float
  = "caml_int64_float_of_bits" "caml_int64_float_of_bits_unboxed"
  [@@unboxed] [@@noalloc]
let infinity =
  float_of_bits 0x7F_F0_00_00_00_00_00_00L
let neg_infinity =
  float_of_bits 0xFF_F0_00_00_00_00_00_00L
let nan =
  float_of_bits 0x7F_F0_00_00_00_00_00_01L
let max_float =
  float_of_bits 0x7F_EF_FF_FF_FF_FF_FF_FFL
let min_float =
  float_of_bits 0x00_10_00_00_00_00_00_00L
let epsilon_float =
  float_of_bits 0x3C_B0_00_00_00_00_00_00L

type fpclass =
    FP_normal
  | FP_subnormal
  | FP_zero
  | FP_infinite
  | FP_nan
external classify_float : (float [@unboxed]) -> fpclass =
  "caml_classify_float" "caml_classify_float_unboxed" [@@noalloc]

(* String and byte sequence operations -- more in modules String and Bytes *)

external string_length : string -> int = "%string_length"
external bytes_length : bytes -> int = "%bytes_length"
external bytes_create : int -> bytes = "caml_create_bytes"
external string_blit : string -> int -> bytes -> int -> int -> unit
                     = "caml_blit_string" [@@noalloc]
external bytes_blit : bytes -> int -> bytes -> int -> int -> unit
                        = "caml_blit_bytes" [@@noalloc]
external bytes_unsafe_to_string : bytes -> string = "%bytes_to_string"
external bytes_unsafe_of_string : string -> bytes = "%bytes_of_string"

let ( ^ ) s1 s2 =
  let l1 = string_length s1 and l2 = string_length s2 in
  let s = bytes_create (l1 + l2) in
  string_blit s1 0 s 0 l1;
  string_blit s2 0 s l1 l2;
  bytes_unsafe_to_string s

(* Character operations -- more in module Char *)

external int_of_char : char -> int = "%identity"
external unsafe_char_of_int : int -> char = "%identity"
let char_of_int n =
  if n < 0 || n > 255 then invalid_arg "char_of_int" else unsafe_char_of_int n

(* Unit operations *)

external ignore : 'a -> unit = "%ignore"

(* Pair operations *)

external fst : 'a * 'b -> 'a = "%field0"
external snd : 'a * 'b -> 'b = "%field1"

(* References *)

type 'a ref = { mutable contents : 'a }
external ref : 'a -> 'a ref = "%makemutable"
external ( ! ) : 'a ref -> 'a = "%field0"
external ( := ) : 'a ref -> 'a -> unit = "%setfield0"
external incr : int ref -> unit = "%incr"
external decr : int ref -> unit = "%decr"

(* Result type *)

type ('a,'b) result = Ok of 'a | Error of 'b

(* String conversion functions *)

external format_int : string -> int -> string = "caml_format_int"
external format_float : string -> float -> string = "caml_format_float"

let string_of_bool b =
  if b then "true" else "false"
let bool_of_string = function
  | "true" -> true
  | "false" -> false
  | _ -> invalid_arg "bool_of_string"

let bool_of_string_opt = function
  | "true" -> Some true
  | "false" -> Some false
  | _ -> None

let string_of_int n =
  format_int "%d" n

external int_of_string : string -> int = "caml_int_of_string"

let int_of_string_opt s =
  (* TODO: provide this directly as a non-raising primitive. *)
  try Some (int_of_string s)
  with Failure _ -> None

external string_get : string -> int -> char = "%string_safe_get"

let valid_float_lexem s =
  let l = string_length s in
  let rec loop i =
    if i >= l then s ^ "." else
    match string_get s i with
    | '0' .. '9' | '-' -> loop (i + 1)
    | _ -> s
  in
  loop 0

let string_of_float f = valid_float_lexem (format_float "%.12g" f)

external float_of_string : string -> float = "caml_float_of_string"

let float_of_string_opt s =
  (* TODO: provide this directly as a non-raising primitive. *)
  try Some (float_of_string s)
  with Failure _ -> None

(* List operations -- more in module List *)

let rec ( @ ) l1 l2 =
  match l1 with
    [] -> l2
  | hd :: tl -> hd :: (tl @ l2)

(* I/O operations *)

type raw_in_channel
type raw_out_channel

external open_descriptor_out : int -> raw_out_channel
                             = "caml_ml_open_descriptor_out"
external open_descriptor_in : int -> raw_in_channel = "caml_ml_open_descriptor_in"

type in_channel =
  | IC_raw of raw_in_channel
  | IC_functions of {
      read: bytes -> int -> int -> int;
      read_char: unit -> char;
      close: unit -> unit;
    }

type out_channel =
  | OC_raw of raw_out_channel
  | OC_functions of {
      write: bytes -> int -> int -> unit;
      write_char: char -> unit;
      close: unit -> unit;
      flush: unit -> unit;
    }

let stdin = IC_raw (open_descriptor_in 0)
let stdout = OC_raw (open_descriptor_out 1)
let stderr = OC_raw (open_descriptor_out 2)

(* General output functions *)

type open_flag =
    Open_rdonly | Open_wronly | Open_append
  | Open_creat | Open_trunc | Open_excl
  | Open_binary | Open_text | Open_nonblock

external open_desc : string -> open_flag list -> int -> int = "caml_sys_open"

external set_out_channel_name: raw_out_channel -> string -> unit =
  "caml_ml_set_channel_name"

let open_out_gen mode perm name : out_channel =
  let c = open_descriptor_out(open_desc name mode perm) in
  set_out_channel_name c name;
  OC_raw c

let open_out name : out_channel =
  open_out_gen [Open_wronly; Open_creat; Open_trunc; Open_text] 0o666 name

let open_out_bin name : out_channel =
  open_out_gen [Open_wronly; Open_creat; Open_trunc; Open_binary] 0o666 name

external raw_flush : raw_out_channel -> unit = "caml_ml_flush"

let flush = function
  | OC_raw c -> raw_flush c
  | OC_functions r -> r.flush()

external raw_out_channels_list : unit -> raw_out_channel list
                           = "caml_ml_out_channels_list"

let flush_all () =
  let rec iter = function
      [] -> ()
    | a::l ->
        begin try
            raw_flush a
        with Sys_error _ ->
          () (* ignore channels closed during a preceding flush. *)
        end;
        iter l
  in iter (raw_out_channels_list ())

external raw_unsafe_output : raw_out_channel -> bytes -> int -> int -> unit
                       = "caml_ml_output_bytes"
external raw_unsafe_output_string : raw_out_channel -> string -> int -> int -> unit
                              = "caml_ml_output"

external raw_output_char : raw_out_channel -> char -> unit = "caml_ml_output_char"

let output_char oc c = match oc with
  | OC_raw oc -> raw_output_char oc c
  | OC_functions r -> r.write_char c

let output_bytes oc s =
  match oc with
  | OC_raw oc -> raw_unsafe_output oc s 0 (bytes_length s)
  | OC_functions r -> r.write s 0 (bytes_length s)

let output_string oc s =
  match oc with
  | OC_raw oc -> raw_unsafe_output_string oc s 0 (string_length s)
  | OC_functions r -> r.write (bytes_unsafe_of_string s) 0 (string_length s)

let output oc s ofs len =
  match oc with
  | OC_raw oc ->
    if ofs < 0 || len < 0 || ofs > bytes_length s - len
    then invalid_arg "output"
    else raw_unsafe_output oc s ofs len
  | OC_functions r -> r.write s ofs len

let output_substring oc s ofs len =
  match oc with
  | OC_raw oc ->
    if ofs < 0 || len < 0 || ofs > string_length s - len
    then invalid_arg "output_substring"
    else raw_unsafe_output_string oc s ofs len
  | OC_functions r ->
    r.write (bytes_unsafe_of_string s) ofs len

external raw_output_byte : raw_out_channel -> int -> unit = "caml_ml_output_char"

let output_byte oc c = match oc with
  | OC_raw oc -> raw_output_byte oc c
  | OC_functions r ->
    let c = char_of_int (c land 0xff) in
    r.write_char c

external raw_output_binary_int : raw_out_channel -> int -> unit = "caml_ml_output_int"

let output_binary_int oc i = match oc with
  | OC_raw oc -> raw_output_binary_int oc i
  | OC_functions r ->
    assert false
(* FIXME: use a Buffer or Bytes function? *)

external raw_marshal_to_channel : raw_out_channel -> 'a -> unit list -> unit
     = "caml_output_value"
external raw_marshal_to_bytes : 'a -> unit list -> bytes
    = "caml_output_value_to_bytes"

let output_value chan v = match chan with
  | OC_raw chan -> raw_marshal_to_channel chan v []
  | OC_functions r ->
    let buf = raw_marshal_to_bytes v [] in
    r.write buf 0 (bytes_length buf)

external raw_seek_out : raw_out_channel -> int -> unit = "caml_ml_seek_out"
external raw_pos_out : raw_out_channel -> int = "caml_ml_pos_out"
external raw_out_channel_length : raw_out_channel -> int = "caml_ml_channel_size"
external raw_close_out_channel : raw_out_channel -> unit = "caml_ml_close_channel"

let seek_out oc i = match oc with
  | OC_raw oc -> raw_seek_out oc i
  | OC_functions _ -> ()

let pos_out = function
  | OC_raw oc -> raw_pos_out oc
  | OC_functions _ -> 0

let out_channel_length = function
  | OC_raw oc -> raw_out_channel_length oc
  | OC_functions _ -> -1

let close_out = function
  | OC_raw oc -> raw_flush oc; raw_close_out_channel oc
  | OC_functions r -> r.flush(); r.close()

let close_out_noerr = function
  | OC_raw oc ->
    (try raw_flush oc with _ -> ());
    (try raw_close_out_channel oc with _ -> ())
  | OC_functions r ->
    (try r.flush () with _ -> ());
    (try r.close() with _ -> ())

external raw_set_binary_mode_out : raw_out_channel -> bool -> unit
                             = "caml_ml_set_binary_mode"

let set_binary_mode_out oc b =
  match oc with
  | OC_raw oc -> raw_set_binary_mode_out oc b
  | OC_functions _ -> ()

(* General input functions *)

external set_in_channel_name: raw_in_channel -> string -> unit =
  "caml_ml_set_channel_name"

let open_in_gen mode perm name =
  let c = open_descriptor_in(open_desc name mode perm) in
  set_in_channel_name c name;
  IC_raw c

let open_in name =
  open_in_gen [Open_rdonly; Open_text] 0 name

let open_in_bin name =
  open_in_gen [Open_rdonly; Open_binary] 0 name

external raw_input_char : raw_in_channel -> char = "caml_ml_input_char"

let input_char = function
  | IC_raw ic -> raw_input_char ic
  | IC_functions r -> r.read_char()

external unsafe_input : in_channel -> bytes -> int -> int -> int
                      = "caml_ml_input"

let input ic s ofs len =
  if ofs < 0 || len < 0 || ofs > bytes_length s - len
  then invalid_arg "input"
  else unsafe_input ic s ofs len

let rec unsafe_really_input ic s ofs len =
  if len <= 0 then () else begin
    let r = unsafe_input ic s ofs len in
    if r = 0
    then raise End_of_file
    else unsafe_really_input ic s (ofs + r) (len - r)
  end

let really_input ic s ofs len =
  if ofs < 0 || len < 0 || ofs > bytes_length s - len
  then invalid_arg "really_input"
  else unsafe_really_input ic s ofs len

let really_input_string ic len =
  let s = bytes_create len in
  really_input ic s 0 len;
  bytes_unsafe_to_string s

external input_scan_line : in_channel -> int = "caml_ml_input_scan_line"

let input_line chan =
  let rec build_result buf pos = function
    [] -> buf
  | hd :: tl ->
      let len = bytes_length hd in
      bytes_blit hd 0 buf (pos - len) len;
      build_result buf (pos - len) tl in
  let rec scan accu len =
    let n = input_scan_line chan in
    if n = 0 then begin                   (* n = 0: we are at EOF *)
      match accu with
        [] -> raise End_of_file
      | _  -> build_result (bytes_create len) len accu
    end else if n > 0 then begin          (* n > 0: newline found in buffer *)
      let res = bytes_create (n - 1) in
      ignore (unsafe_input chan res 0 (n - 1));
      ignore (input_char chan);           (* skip the newline *)
      match accu with
        [] -> res
      |  _ -> let len = len + n - 1 in
              build_result (bytes_create len) len (res :: accu)
    end else begin                        (* n < 0: newline not found *)
      let beg = bytes_create (-n) in
      ignore(unsafe_input chan beg 0 (-n));
      scan (beg :: accu) (len - n)
    end
  in bytes_unsafe_to_string (scan [] 0)

external raw_input_byte : raw_in_channel -> int = "caml_ml_input_char"
external raw_input_binary_int : raw_in_channel -> int = "caml_ml_input_int"
external raw_input_value : raw_in_channel -> 'a = "caml_input_value"
external raw_seek_in : raw_in_channel -> int -> unit = "caml_ml_seek_in"
external raw_pos_in : raw_in_channel -> int = "caml_ml_pos_in"
external raw_in_channel_length : raw_in_channel -> int = "caml_ml_channel_size"
external close_in : raw_in_channel -> unit = "caml_ml_close_channel"
let close_in_noerr ic = (try close_in ic with _ -> ())
external set_binary_mode_in : raw_in_channel -> bool -> unit
                            = "caml_ml_set_binary_mode"

let input_byte = function
  | IC_raw ic -> int_of_char (raw_input_byte ic)
  | IC_functions r -> r.read_char()

(* Output functions on standard output *)

let print_char c = output_char stdout c
let print_string s = output_string stdout s
let print_bytes s = output_bytes stdout s
let print_int i = output_string stdout (string_of_int i)
let print_float f = output_string stdout (string_of_float f)
let print_endline s =
  output_string stdout s; output_char stdout '\n'; flush stdout
let print_newline () = output_char stdout '\n'; flush stdout

(* Output functions on standard error *)

let prerr_char c = output_char stderr c
let prerr_string s = output_string stderr s
let prerr_bytes s = output_bytes stderr s
let prerr_int i = output_string stderr (string_of_int i)
let prerr_float f = output_string stderr (string_of_float f)
let prerr_endline s =
  output_string stderr s; output_char stderr '\n'; flush stderr
let prerr_newline () = output_char stderr '\n'; flush stderr

(* Input functions on standard input *)

let read_line () = flush stdout; input_line stdin
let read_int () = int_of_string(read_line())
let read_int_opt () = int_of_string_opt(read_line())
let read_float () = float_of_string(read_line())
let read_float_opt () = float_of_string_opt(read_line())

(* Operations on large files *)

module LargeFile =
  struct
    external raw_seek_out : raw_out_channel -> int64 -> unit = "caml_ml_seek_out_64"
    external raw_pos_out : raw_out_channel -> int64 = "caml_ml_pos_out_64"
    external raw_out_channel_length : raw_out_channel -> int64
                                = "caml_ml_channel_size_64"

    let seek_out oc i = match oc with
      | OC_raw oc -> raw_seek_out oc i
      | OC_functions _ -> ()

    let pos_out = function
      | OC_raw oc -> raw_pos_out oc
      | OC_functions _ -> 0L

    let out_channel_length = function
      | OC_raw oc -> raw_out_channel_length oc
      | OC_functions _ -> -1L

    external raw_seek_in : raw_in_channel -> int64 -> unit = "caml_ml_seek_in_64"
    external raw_pos_in : raw_in_channel -> int64 = "caml_ml_pos_in_64"
    external raw_in_channel_length : raw_in_channel -> int64 = "caml_ml_channel_size_64"


  end

(* Formats *)

type ('a, 'b, 'c, 'd, 'e, 'f) format6
   = ('a, 'b, 'c, 'd, 'e, 'f) CamlinternalFormatBasics.format6
   = Format of ('a, 'b, 'c, 'd, 'e, 'f) CamlinternalFormatBasics.fmt
               * string

type ('a, 'b, 'c, 'd) format4 = ('a, 'b, 'c, 'c, 'c, 'd) format6

type ('a, 'b, 'c) format = ('a, 'b, 'c, 'c) format4

let string_of_format (Format (_fmt, str)) = str

external format_of_string :
 ('a, 'b, 'c, 'd, 'e, 'f) format6 ->
 ('a, 'b, 'c, 'd, 'e, 'f) format6 = "%identity"

let ( ^^ ) (Format (fmt1, str1)) (Format (fmt2, str2)) =
  Format (CamlinternalFormatBasics.concat_fmt fmt1 fmt2,
          str1 ^ "%," ^ str2)

(* Miscellaneous *)

external sys_exit : int -> 'a = "caml_sys_exit"

let exit_function = CamlinternalAtomic.make flush_all

let rec at_exit f =
  let module Atomic = CamlinternalAtomic in
  (* MPR#7253, MPR#7796: make sure "f" is executed only once *)
  let f_yet_to_run = Atomic.make true in
  let old_exit = Atomic.get exit_function in
  let new_exit () =
    if Atomic.compare_and_set f_yet_to_run true false then f () ;
    old_exit ()
  in
  let success = Atomic.compare_and_set exit_function old_exit new_exit in
  if not success then at_exit f

let do_at_exit () = (CamlinternalAtomic.get exit_function) ()

let exit retcode =
  do_at_exit ();
  sys_exit retcode

let _ = register_named_value "Pervasives.do_at_exit" do_at_exit

external major : unit -> unit = "caml_gc_major"
external naked_pointers_checked : unit -> bool
  = "caml_sys_const_naked_pointers_checked"
let () = if naked_pointers_checked () then at_exit major

(*MODULE_ALIASES*)
module Arg          = Arg
module Array        = Array
module ArrayLabels  = ArrayLabels
module Atomic       = Atomic
module Bigarray     = Bigarray
module Bool         = Bool
module Buffer       = Buffer
module Bytes        = Bytes
module BytesLabels  = BytesLabels
module Callback     = Callback
module Char         = Char
module Complex      = Complex
module Digest       = Digest
module Either       = Either
module Ephemeron    = Ephemeron
module Filename     = Filename
module Float        = Float
module Format       = Format
module Fun          = Fun
module Gc           = Gc
module Genlex       = Genlex
module Hashtbl      = Hashtbl
module Int          = Int
module Int32        = Int32
module Int64        = Int64
module Lazy         = Lazy
module Lexing       = Lexing
module List         = List
module ListLabels   = ListLabels
module Map          = Map
module Marshal      = Marshal
module MoreLabels   = MoreLabels
module Nativeint    = Nativeint
module Obj          = Obj
module Oo           = Oo
module Option       = Option
module Parsing      = Parsing
module Pervasives   = Pervasives
module Printexc     = Printexc
module Printf       = Printf
module Queue        = Queue
module Random       = Random
module Result       = Result
module Scanf        = Scanf
module Seq          = Seq
module Set          = Set
module Stack        = Stack
module StdLabels    = StdLabels
module Stream       = Stream
module String       = String
module StringLabels = StringLabels
module Sys          = Sys
module Uchar        = Uchar
module Unit         = Unit
module Weak         = Weak
