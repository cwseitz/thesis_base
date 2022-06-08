from distutils.core import setup, Extension
import numpy

def main():
    
    setup(name="thesis_base",
          version="1.0.0",
          description="Library code for my PhD thesis",
          author="Clayton Seitz",
          author_email="cwseitz@iu.edu",
          ext_modules=[Extension("thesis_base.backend", ["thesis_base/backend/backend.c"],
                       include_dirs = [numpy.get_include()],
                       library_dirs = ['/usr/lib/x86_64-linux-gnu'])])



if __name__ == "__main__":
    main()
