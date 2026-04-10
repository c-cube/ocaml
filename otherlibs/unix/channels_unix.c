/**************************************************************************/
/*                                                                        */
/*                                 OCaml                                  */
/*                                                                        */
/*             Xavier Leroy, projet Gallium, INRIA Paris                  */
/*                                                                        */
/*   Copyright 2017 Institut National de Recherche en Informatique et     */
/*     en Automatique.                                                    */
/*                                                                        */
/*   All rights reserved.  This file is distributed under the terms of   */
/*   the GNU Lesser General Public License version 2.1, with the         */
/*   special exception on linking described in the file LICENSE.          */
/*                                                                        */
/**************************************************************************/

#define CAML_INTERNALS

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include "caml/unixsupport.h"

#ifdef HAS_SOCKETS
#include <sys/socket.h>
#include "caml/socketaddr.h"
#endif

/* Check that the given file descriptor has "stream semantics" and
   can therefore be used as part of buffered I/O.  Things that
   don't have "stream semantics" include block devices and
   UDP (datagram) sockets.
   Raises Unix_error on failure, returns Val_unit on success. */

CAMLprim value caml_unix_check_stream_semantics(value vfd, value vcall)
{
  struct stat buf;
  int fd = Int_val(vfd);

  caml_enter_blocking_section();
  if (fstat(fd, &buf) == -1) {
    int err = errno;
    caml_leave_blocking_section();
    caml_unix_error(err, String_val(vcall), Nothing);
  }
  caml_leave_blocking_section();

  switch (buf.st_mode & S_IFMT) {
  case S_IFREG: case S_IFCHR: case S_IFIFO:
    /* These have stream semantics */
    return Val_unit;
#ifdef HAS_SOCKETS
  case S_IFSOCK: {
    int so_type;
    socklen_param_type so_type_len = sizeof(so_type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &so_type, &so_type_len) == -1)
      caml_unix_error(errno, String_val(vcall), Nothing);
    switch (so_type) {
    case SOCK_STREAM:
      return Val_unit;
    default:
      caml_unix_error(EINVAL, String_val(vcall), Nothing);
    }
    }
#endif
  default:
    /* All other file types are suspect: block devices, directories,
       symbolic links, whatnot. */
    caml_unix_error(EINVAL, String_val(vcall), Nothing);
  }
}
