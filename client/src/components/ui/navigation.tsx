import { Link, useLocation } from "wouter";
import { CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export function Navigation() {
  const [location] = useLocation();

  return (
    <nav className="bg-white shadow-sm border-b border-slate-200">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-semibold text-slate-900">OSHA Compliance</h1>
          </Link>
          <div className="flex items-center space-x-6">
            <Link 
              href="/"
              className={cn(
                "font-medium transition-colors duration-200",
                location === "/" 
                  ? "text-slate-700" 
                  : "text-slate-500 hover:text-primary"
              )}
            >
              Home
            </Link>
            <Link 
              href="/about"
              className={cn(
                "font-medium transition-colors duration-200",
                location === "/about" 
                  ? "text-slate-700" 
                  : "text-slate-500 hover:text-primary"
              )}
            >
              About
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
