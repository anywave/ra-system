{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -Wno-missing-safe-haskell-mode #-}
module Paths_ra_system (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\bin"
libdir     = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-9.0.1\\ra-system-0.1.0.0-inplace-ra-system-test"
dynlibdir  = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-9.0.1"
datadir    = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-9.0.1\\ra-system-0.1.0.0"
libexecdir = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\ra-system-0.1.0.0-inplace-ra-system-test\\x86_64-windows-ghc-9.0.1\\ra-system-0.1.0.0"
sysconfdir = "C:\\Users\\schmi\\AppData\\Roaming\\cabal\\etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "ra_system_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "ra_system_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "ra_system_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "ra_system_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "ra_system_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "ra_system_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "\\" ++ name)
