# Cerebrum

Fast and flexible Deep Learning platform written in C++

The platform is under **intensive** development. Version 0.1 is scheduled for
Thursday, 23rd of April. Stay tuned!

## Using BLAS

Cerebrum can use BLAS to speed up computation. BLAS is not a mandatory
dependency, but we recommend you to use it.

You can use any CBLAS implementation, (e.g.
 [ATLAS](http://math-atlas.sourceforge.net/),
 [OpenBLAS](www.openblas.net)), but you have to let the compiler know about
the specific flags and the location of the libraries to be linked.

### Compiling with BLAS

You have two options to send the compiler the flags needed in order to use
BLAS.

#### Using configuration files

1.  Create a file named `cblas.cflags` with the needed pre-processor, and
    compiler flags. For example, `cblas.c` might contain the following line:
    
    ```
    -I/usr/include/openblas/
    ```

2.  Create a file named `cblas.libs` with the needed linker flags. For example,
    `cblas.libs` might have the following contents:
    
    ```
    -L/usr/lib64/ -lopenblas -lpthread -lgfortran
    ```

3.  Add `cblas` to the goals you send to `make`:
    
    ```
    $ make cblas test_cblas
    $ ./test_cblas
    ok
    ```

###### Specify the BLAS implementation

1.  Make sure that `pkg-config` provides the needed pre-processor, compiler,
    and linker flags. For example:
    
    ```
    $ pkg-config atlas --cflags --libs
    -I/usr/include/atlas/ -L/usr/lib64/atlas/ -lsatlas
    ```

2. Send the package name as a goal to the `make` command:
    
    ```
    $ make atlas test_cblas
    $ ./test_cblas
    ok
    ```

### Check that BLAS works

1.  Compile `test_cblas` using one of the alternatives described above:
    
    ```
    $ make cblas test_cblas
    ```
    or
    ```
    $ make atlas test_cblas
    ```

2.  Run the program:
    
    ```
    $ ./test_cblas
    ok
    ```

