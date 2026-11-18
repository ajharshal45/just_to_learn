import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Problem from "./components/Problem";
import Algorithms from "./components/Algorithms";
import Simulator from "./components/Simulator";
import Pipeline from "./components/Pipeline";
import LeakageProof from "./components/LeakageProof";
import BenchmarkTable from "./components/BenchmarkTable";
import Charts from "./components/Charts";
import PaperSection from "./components/PaperSection";
import CodeSnippet from "./components/CodeSnippet";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen bg-dark text-slate-100">
      <Navbar />
      <main>
        <Hero />
        <Problem />
        <Algorithms />
        <Simulator />
        <Pipeline />
        <LeakageProof />
        <BenchmarkTable />
        <Charts />
        <PaperSection />
        <CodeSnippet />
      </main>
      <Footer />
    </div>
  );
}
